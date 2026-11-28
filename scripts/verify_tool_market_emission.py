"""Read a signed SN71 target and actual miner emissions at the first finalized distribution after its three validator reveals.

python -m scripts.verify_tool_market_emission --state state.json --signing-key public-key.json
Pass --control for an older state without a tool-market allocation. No credentials, writes or signing
authority are needed. The archive authenticates chain observations. The operator signature
authenticates the supplied target. Neither establishes a USD payment for a particular tool call.
Exit 2 means the observation contains an unmatched target or a requested reveal lacks proof.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from Leadpoet.utils.subnet_epoch import OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, read_subnet_epoch_snapshot
from lab_arena import chain, signing
from lab_arena.collection_distributions import miner_events, validator_vectors
from lab_arena.chain_source import ValidatorChainSourceV2
from lab_arena.local_weight_signer import HttpsJsonRpcTransport, load_public_chain_signing_profile
from lab_arena.validator import FINNEY_SN71_SIGNING_KEY_HASH
from leadpoet_canonical.arena_weights import derive_arena_weights, verify_accepted_weight_state_signature
from leadpoet_canonical.chain_source_v2 import (
    decode_timelocked_weight_commits, timelocked_weight_commits_storage_key,
)

VALIDATORS = {
    "primary": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
    "Rizzo": "5GeEtY7Ars1FbSKhueoffzLHrYgw3rxn7eF6egicCbWmNs71",
    "Yuma": "5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW",
}


def compare_registries(before, after):
    if (not before.hotkeys or len(before.hotkeys) != len(before.coldkeys)
            or len(set(before.hotkeys)) != len(before.hotkeys)
            or before.hotkeys != after.hotkeys or before.coldkeys != after.coldkeys):
        raise ValueError("UID or owner registry differs at distribution; resolve event order independently")


def recipients(state, graph, weights, amounts):
    market = state.get("tool_market", {}).get("allocations_microusd", {})
    lab = state["reward_basis"].get("quotes_microusd", {})
    rows = []
    for uid, (hotkey, coldkey) in enumerate(zip(graph.hotkeys, graph.coldkeys)):
        if amounts[uid] or uid in weights or hotkey in market or hotkey in lab:
            rows.append({
                "uid": uid, "hotkey": hotkey, "coldkey": coldkey, "signed_weight_u16": weights.get(uid, 0),
                "market_target_microusd": market.get(hotkey, 0), "lab_quote_microusd": lab.get(hotkey, 0),
                "designated_burn": hotkey == state["burn_hotkey"], "observed_miner_alpha_rao": amounts[uid],
            })
    return rows


def closed_boundary(graph, boundary, cutover):
    if type(boundary) is not int or boundary <= 0:
        raise ValueError("invalid distribution block")
    head = graph.finalized_head()
    parent, block = (graph.finalized_block_hash(n, head=head) for n in (boundary - 1, boundary))
    prev, current = (read_subnet_epoch_snapshot(graph.epoch_subtensor(), netuid=71, block_hash=h) for h in (parent, block))
    if (prev.current_block != boundary - 1 or current.current_block != boundary or current.last_epoch_block != boundary
            or current.subnet_epoch_index != prev.subnet_epoch_index + 1
            or current.network_genesis_hash != prev.network_genesis_hash
            or current.network_genesis_hash != cutover.network_genesis_hash):
        raise ValueError("block does not close a finalized Finney SN71 epoch")
    return parent, block, prev, current


def prove_votes(rpc, cutover, signed, target_boundary, parent, target_epoch, registry, expected):
    """Recover the signed epoch's queued commitments and prove their reveal events and weights."""
    from bittensor_wallet import Keypair

    source = ValidatorChainSourceV2(
        rpc_call=rpc.call, archive_rpc_call=rpc.call,
        epoch_authority_supplier=lambda: {"mode": "stateful_v1", "cutover_manifest": cutover.to_dict()},
    )
    profile = load_public_chain_signing_profile("finney")
    index = target_epoch.subnet_epoch_index
    key = timelocked_weight_commits_storage_key(netuid=71, subnet_epoch_index=index)
    queue = decode_timelocked_weight_commits(rpc.call(method="state_getStorage", params=[key, parent], request_id=3))
    proofs = {}
    for name, hotkey in VALIDATORS.items():
        public = bytes(Keypair(ss58_address=hotkey).public_key).hex()
        records = [e for e in queue if e["hotkey_public_key"] == public]
        if len(records) != 1:
            raise ValueError(f"expected one {name} commitment in the signed epoch's closing queue")
        record = records[0]
        if not signed["valid_from_block"] <= record["submitted_at"] < target_boundary:
            raise ValueError("queued commitment outside signed epoch")
        proof = source.prove_timelocked_reveal_transition(
            netuid=71, validator_hotkey=hotkey, hotkey_public_key_hex=public, subnet_epoch_index=index,
            commitment_hex=record["commitment_hex"], reveal_round=record["reveal_round"],
            inclusion_block=record["submitted_at"],
            reveal_deadline_block=target_boundary + profile["subnet_reveal_period_epochs"] * target_epoch.tempo,
            expected_weights=expected, expected_recipient_uid_hotkeys=[
                {"uid": uid, "hotkey": registry.hotkeys[uid]} for uid, _ in expected], chain_profile=profile,
        )
        if proof is None:
            raise ValueError(f"{name} reveal pending or unproved")
        proofs[name] = {
            "hotkey": hotkey, "uid": registry.hotkeys.index(hotkey),
            "commit_block": record["submitted_at"], "reveal_block": proof["reveal_block"],
            "reveal_hash": proof["reveal_block_hash"], "transition_hash": proof["transition_hash"],
        }
    return proofs


def verify_receipt(graph, rpc, state, key_doc, *, control=False, boundary=None):
    signature = verify_accepted_weight_state_signature(
        state, public_key_der=signing.load_public_key_from_document(key_doc),
        expected_public_key_hash=FINNEY_SN71_SIGNING_KEY_HASH,
    )
    if ("tool_market" in state) == control:
        raise ValueError("expected a signed market allocation, or --control with a historical non-market state")
    cutover = chain.load_arena_cutover({})
    target_boundary = state["valid_until_block"] + 1
    target_parent, _, target_epoch, _ = closed_boundary(graph, target_boundary, cutover)
    if (state["network"] != "finney" or state["netuid"] != 71
            or state["genesis_hash"] != cutover.network_genesis_hash.removeprefix("0x")
            or state["epoch"] != target_epoch.settlement_epoch_id(cutover)
            or datetime.fromisoformat(state["issued_at"].replace("Z", "+00:00")) >
               datetime.fromisoformat(target_epoch.observed_at.replace("Z", "+00:00"))):
        raise ValueError("signed target belongs to another chain or epoch")
    read_registry = chain.bittensor_metagraph_source(graph.config)
    registry = read_registry(graph.client, 71, target_parent)
    target = derive_arena_weights(state, registry.hotkeys)
    expected = list(zip(target["sparse_uids"], target["sparse_weights_u16"]))
    proofs = prove_votes(rpc, cutover, state, target_boundary, target_parent, target_epoch, registry, expected)
    latest = max(v["reveal_block"] for v in proofs.values())
    latest_hash = graph.finalized_block_hash(latest)
    revealed = read_subnet_epoch_snapshot(graph.epoch_subtensor(), netuid=71, block_hash=latest_hash)
    boundary = revealed.next_epoch_block if boundary is None else boundary
    parent, block, prev, current = closed_boundary(graph, boundary, cutover)
    if boundary <= latest or prev.subnet_epoch_index != revealed.subnet_epoch_index:
        raise ValueError("expected the first distribution strictly after all three verified reveals")

    before, after = (read_registry(graph.client, 71, h) for h in (parent, block))
    compare_registries(registry, before)
    compare_registries(before, after)
    for label, h in (("parent", parent), ("boundary", block)):
        checks = validator_vectors(rpc, h, before.hotkeys, state["epoch"], target["weights_hash"], VALIDATORS)
        for name, verdict in checks.items():
            proofs[name][label + "_matches_target"] = verdict == "matches"
    events = miner_events(graph.client.get_events(block), {len(before.hotkeys)})
    index, amounts = events[0]
    targets = dict(expected)
    rows = recipients(state, before, targets, amounts)
    market = state.get("tool_market", {}).get("allocations_microusd", {})
    return {
        "schema": "leadpoet.tool_market.miner_emission_observation.v1",
        "mode": "control" if control else "signed_market_target", "network": "finney", "netuid": 71,
        "signed_epoch": state["epoch"], "distribution_closed_epoch": prev.settlement_epoch_id(cutover),
        "state_hash": signature, "target_weights_hash": target["weights_hash"],
        "distribution_block": boundary, "block_hash": block, "parent_hash": parent,
        "observed_at": current.observed_at, "event_index": index, "event_phase": "Initialization",
        "other_mechanism_alpha_rao": {str(i): sum(values) for i, (_, values) in events.items() if i != 0},
        "mechanism_zero_total_miner_alpha_rao": sum(amounts), "recipients": rows,
        "unregistered_supplier_hotkeys": sorted(set(market) - set(before.hotkeys)),
        "validators": proofs,
        "all_validator_vectors_match": all(v["parent_matches_target"] and v["boundary_matches_target"]
                                           for v in proofs.values()),
        "unexpected_positive_uids": [uid for uid, amount in enumerate(amounts) if amount and uid not in targets],
        "zero_emission_supplier_hotkeys": [r["hotkey"] for r in rows
                                           if r["market_target_microusd"] and not r["observed_miner_alpha_rao"]],
        "meaning": "Observed miner incentive for each UID from all consensus inputs. Micro-USD columns are signed "
                   "targets. Coldkeys are registered owners at distribution. This observation does not prove a liquid "
                   "wallet credit, attribute a portion of blended incentives to an invoice, or count them as paid USD.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path, help="Signed Finney accepted-weight-state JSON document")
    parser.add_argument("--signing-key", required=True, type=Path, help="Published Arena signing-key JSON document")
    parser.add_argument("--control", action="store_true", help="Only accept a real signed state without a Gamma allocation")
    parser.add_argument("--distribution-block", type=int, help="Override the predicted next boundary; the archive must "
                        "verify this is the first actual distribution strictly after all three reveals")
    args = parser.parse_args()
    cfg = chain.ArenaChainConfig(OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, 71, "finney", 20)
    graph = chain.ArenaChain(cfg, chain.connect_substrate(cfg))
    transport = HttpsJsonRpcTransport(OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, timeout_seconds=20)
    try:
        receipt = verify_receipt(graph, transport, json.loads(args.state.read_bytes()),
                                 json.loads(args.signing_key.read_bytes()), control=args.control,
                                 boundary=args.distribution_block)
        print(json.dumps(receipt, sort_keys=True))
        if (not receipt["all_validator_vectors_match"] or receipt["unregistered_supplier_hotkeys"]
                or receipt["zero_emission_supplier_hotkeys"] or receipt["unexpected_positive_uids"]):
            raise SystemExit(2)
    finally:
        graph.close()
        transport.close()


if __name__ == "__main__":
    main()
