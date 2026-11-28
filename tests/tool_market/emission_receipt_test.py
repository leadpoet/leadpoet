from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from lab_arena import signing, weight_state
from leadpoet_canonical.arena_weights import ArenaWeightError, derive_arena_weights
from scripts import verify_tool_market_emission as receipt
from tests.lab_arena.collection_rewards_test import BURN, EPOCH, GENESIS, MINER, SECOND, SIGNER, state


def event(amounts, index=71):
    return {"phase": "Initialization", "event": {"module_id": "SubtensorModule",
            "event_id": "IncentiveAlphaEmittedToMiners", "attributes": {"netuid": index, "emissions": amounts}}}


def test_ignores_other_mechanisms_and_rejects_ambiguous_events():
    record = event([1, 2])
    assert receipt.miner_events([event([3, 4], 4096 + 71), record], {2})[0] == (1, [1, 2])
    for malformed in ([record, record], [], [event([True, 2])], [event([1])], [event([1, -1])], [event([1, 2], 71.0)]):
        with pytest.raises(ValueError):
            receipt.miner_events(malformed, {2})


def signed_target():
    historical = state({MINER: 250_000})
    return weight_state.build_accepted_weight_state(
        SIGNER, network="finney", genesis_hash=GENESIS, netuid=71, epoch=EPOCH,
        valid_from_block=50, valid_until_block=409, reward_basis=historical["reward_basis"],
        burn_hotkey=BURN, issued_at=historical["issued_at"], tool_market={
            "schema_version": "leadpoet.tool_market.supplier_allocations.v1", "epoch": EPOCH,
            "emission_microusd": 1_000_000, "allocations_microusd": {MINER: 250_000}},
    )


def test_real_signature_epoch_scope_event_and_shared_recipient(monkeypatch):
    doc = signed_target()
    signing_key = signing.signing_key_document(SIGNER.public_key_der)
    monkeypatch.setattr(receipt, "FINNEY_SN71_SIGNING_KEY_HASH", SIGNER.public_key_hash)
    cutover = NS(network_genesis_hash=GENESIS, to_dict=lambda: {})
    monkeypatch.setattr(receipt.chain, "load_arena_cutover", lambda env: cutover)
    heights = {409: "target_parent", 410: "target_boundary", 415: "last_reveal", 769: "payout_parent", 770: "payout"}
    graphs = {h: NS(hotkeys=(BURN, MINER, SECOND), coldkeys=(SECOND, BURN, MINER)) for h in heights.values()}

    def snap(n):
        subnet_epoch = 99 if n < 410 else 100 if n < 770 else 101
        return NS(network_genesis_hash=GENESIS, current_block=n, last_epoch_block={99: 50, 100: 410, 101: 770}[subnet_epoch],
                  subnet_epoch_index=subnet_epoch, tempo=360, next_epoch_block=770,
                  settlement_epoch_id=lambda c: EPOCH + subnet_epoch - 99, observed_at="2026-11-24T00:00:00Z")

    snapshots = {h: snap(n) for n, h in heights.items()}
    monkeypatch.setattr(receipt, "read_subnet_epoch_snapshot", lambda client, netuid, block_hash: snapshots[block_hash])
    monkeypatch.setattr(receipt.chain, "bittensor_metagraph_source", lambda cfg: lambda client, netuid, h: graphs[h])
    monkeypatch.setattr(receipt, "VALIDATORS", {"primary": BURN, "Rizzo": MINER, "Yuma": SECOND})
    w = derive_arena_weights(doc, graphs["payout"].hotkeys)
    expected = list(zip(w["sparse_uids"], w["sparse_weights_u16"]))
    from lab_arena import collection_distributions
    from bittensor_wallet import Keypair
    queue = [dict(hotkey_public_key=bytes(Keypair(ss58_address=k).public_key).hex(), submitted_at=55,
                  reveal_round=234, commitment_hex="ff") for k in graphs["payout"].hotkeys]
    monkeypatch.setattr(collection_distributions, "decode_weights_storage", lambda raw: raw)
    monkeypatch.setattr(receipt, "decode_timelocked_weight_commits", lambda raw: queue)
    def prove(**args):
        assert args["expected_weights"] == expected
        assert args["inclusion_block"] == 55
        return {"reveal_block": 415, "reveal_block_hash": "last_reveal", "transition_hash": "hash"}
    monkeypatch.setattr(receipt, "ValidatorChainSourceV2", lambda **args: NS(prove_timelocked_reveal_transition=prove))
    g = NS(config=None, finalized_head=lambda: NS(number=775), finalized_block_hash=lambda n, head=None: heights[n],
           epoch_subtensor=lambda: None, client=NS(get_events=lambda h: [event([5, 8, 3])] if h == "payout" else []))
    def read(target=doc, **kwargs):
        return receipt.verify_receipt(g, NS(call=lambda **x: expected), target, signing_key, **kwargs)

    got = read()
    assert got["all_validator_vectors_match"]
    assert got["distribution_block"] == 770 and got["signed_epoch"] == EPOCH and got["distribution_closed_epoch"] == EPOCH + 1
    assert len(got["validators"]) == 3 and all(v["reveal_block"] == 415 for v in got["validators"].values())
    row = next(row for row in got["recipients"] if row["hotkey"] == MINER)
    assert row["coldkey"] == BURN and row["uid"] == 1
    assert row["market_target_microusd"] == row["lab_quote_microusd"] == 250_000
    assert row["observed_miner_alpha_rao"] == 8 and not got["zero_emission_supplier_hotkeys"]
    assert got["unexpected_positive_uids"] == [2]
    assert sum(row["observed_miner_alpha_rao"] for row in got["recipients"]) == got["mechanism_zero_total_miner_alpha_rao"] == 16

    with pytest.raises(ValueError, match="market allocation"):
        read(state({MINER: 100_000}))
    with pytest.raises(ValueError, match="market allocation"):
        read(control=True)
    with pytest.raises(ValueError, match="strictly after"):
        read(boundary=410)
    forged = deepcopy(doc)
    forged["tool_market"]["allocations_microusd"][MINER] += 1
    with pytest.raises(ArenaWeightError):
        read(forged)
    graphs["payout"].coldkeys = (MINER, BURN, SECOND)
    with pytest.raises(ValueError, match="owner"):
        read()
    graphs["payout"].coldkeys = graphs["payout_parent"].coldkeys
    queue.pop()
    with pytest.raises(ValueError, match="commitment"):
        read()
