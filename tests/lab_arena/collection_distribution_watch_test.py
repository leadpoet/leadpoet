"""Use genuine signed weight states and synthetic chain fixtures to test the observer."""
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from lab_arena import signing, weight_state
from lab_arena import collection_distributions as sampler
from scripts import collection_participation_watch as watch
from leadpoet_canonical.arena_weights import COLLECTION_BASIS_SCHEMA


def test_closed_epoch_event_is_distinct_from_the_current_signed_target():
    def event(netuid, amounts):
        return {"phase": "Initialization", "event": {"module_id":"SubtensorModule",
                "event_id":"IncentiveAlphaEmittedToMiners",
                "attributes":{"netuid":netuid,"emissions":amounts}}}
    events = [event(70,[7,4]), event(71,[90,10]), event(4167,[0,3])]
    actual = sampler.miner_events(events, {2})
    assert actual == {0:(1,[90,10]), 1:(2,[0,3])}
    for bad in (events[:1], [events[1],events[1]], [event(71,[True,1])],
                [event(71,[24])], [dict(events[1],phase="ApplyExtrinsic")]):
        with pytest.raises(ValueError):
            sampler.miner_events(bad,{2})


@pytest.fixture
def evidence(monkeypatch):
    keys = list(watch.VALIDATORS.values())
    signer = signing.LocalSigner.generate()
    snapshots = {"parent": SimpleNamespace(current_block=1359, network_genesis_hash="0x"+"1"*64,
                                           observed_at="2026-11-25T01:00:00Z", settlement_epoch_id=lambda _:100),
                 "close": SimpleNamespace(current_block=1360, last_epoch_block=1360, observed_at="2026-11-25T01:00:12Z",
                                          settlement_epoch_id=lambda _:101)}
    graph = SimpleNamespace(finalized_block_hash=lambda n: {1359:"parent",1360:"close"}[n], epoch_subtensor=lambda:None)
    monkeypatch.setattr(sampler, "read_subnet_epoch_snapshot", lambda _client, netuid, block_hash: snapshots[block_hash])
    matches = dict.fromkeys(watch.VALIDATORS, "matches")
    monkeypatch.setattr(sampler, "validator_vectors", lambda *args: dict(matches))
    config = {"actual": [2000,0,0], "ledger_mismatch": False}
    monkeypatch.setattr(sampler, "registry_at", lambda _rpc, n, h: config.get("new_registry", keys) if h == "close" else keys)
    def signed(epoch):
        basis = {"schema_version": COLLECTION_BASIS_SCHEMA, "source_epoch": epoch-1, "job_count":0,
                 "job_set_sha256":sha256(b"").hexdigest(), "quotes_microusd":{}, "valuation":None}
        snapshot = {"block":1000, "block_hash":"0x"+"2"*64, "miner_alpha_rao":10**9,
                    "tao_rao_per_alpha":10**9, "tao_price_microusd":100, "price_updated_at":1,
                    "observed_at":1, "emission_microusd":100}
        market = None
        if epoch == 100 and config.get("market"):
            market = {"epoch": epoch, "schema_version": "leadpoet.tool_market.supplier_allocations.v1",
                      "emission_microusd": 100, "allocations_microusd": {keys[2]: 25}}
            if not config.get("paid"):
                market["valuation"] = snapshot
        if epoch == 100 and config.get("paid"):
            basis.update(job_count=1, job_set_sha256="b"*64, quotes_microusd={keys[1]:50},
                         valuation=snapshot)
        return weight_state.build_accepted_weight_state(
            signer, network="finney", netuid=71, genesis_hash="1"*64, epoch=epoch,
            valid_from_block=1000, valid_until_block=1359, reward_basis=basis,
            burn_hotkey=keys[0], issued_at="2026-11-25T00:00:00Z", slot_eligibility=None, tool_market=market)
    monkeypatch.setattr(sampler, "ArenaStore", lambda _: SimpleNamespace(
        collection_reward_start_epoch=lambda:100, get_weight_state=lambda _net, _uid, e: {"state_doc":signed(e)}))
    def billing(name, params):
        assert name == "collection_reward_quotes_v1" and params == {"p_epoch":100}
        basis = signed(100)["reward_basis"].copy()
        basis.pop("valuation")
        if config["ledger_mismatch"]:
            basis["job_count"] += 1
        return basis
    graph.client = SimpleNamespace(get_events=lambda _: [
        {"phase":"Initialization","event":{"module_id":"SubtensorModule","event_id":"IncentiveAlphaEmittedToMiners",
                                         "attributes":{"netuid":71,"emissions":config["actual"]}}}])
    return config, matches, keys, lambda: sampler.receipt_at(
        graph, None, SimpleNamespace(rpc=billing), signer, None, 1360, 100, watch.VALIDATORS)


def test_supplier_only_and_mixed_recipients_keep_separate_signed_targets(evidence):
    config, matches, keys, receipt = evidence
    config.update(market=True, actual=[1500, 0, 500])
    row = receipt()
    assert row["result"] == "supplier_recipients_observed"
    assert row["quotes_microusd"] == {} and row["supplier_targets_microusd"] == {keys[2]: 25}
    config.update(paid=True, actual=[500, 1000, 500])
    row = receipt()
    assert row["result"] == "collection_recipients_observed"
    assert row["quotes_microusd"] == {keys[1]: 50} and row["supplier_targets_microusd"] == {keys[2]: 25}
    matches["yuma"] = "differs"
    assert receipt()["result"] == "inspect"


def test_actual_burn_and_paid_recipients_require_ledger_both_vector_checks_and_event(evidence):
    config, matches, keys, receipt = evidence
    row = receipt()
    assert row["result"] == "burn_only" and row["miner_emissions_rao"] == {keys[0]:2000}
    config["actual"] = [1000,1000,0]
    assert receipt()["result"] == "inspect"
    config["paid"] = True
    row = receipt()
    assert row["result"] == "collection_recipients_observed" and row["quotes_microusd"] == {keys[1]:50}
    assert row["miner_emissions_rao"] == {keys[0]:1000,keys[1]:1000}
    config["actual"] = [2000,0,0]
    assert receipt()["result"] == "inspect"
    config["paid"] = False
    matches["yuma"] = "differs"
    assert receipt()["result"] == "inspect"
    matches["yuma"] = "matches"
    config["new_registry"] = [keys[2],keys[1],keys[0]]
    ambiguous = receipt()
    assert ambiguous["result"] == "inspect" and ambiguous["miner_emissions_rao"] == {}
    assert ambiguous["ambiguous_recipient_uids"] == [
        {"uid":0,"parent_hotkey":keys[0],"boundary_hotkey":keys[2],"rao":2000}]
    config["ledger_mismatch"] = True
    with pytest.raises(ValueError, match="accepted jobs"):
        receipt()


def test_epoch_files_are_idempotent_and_missing_epochs_are_processed_in_order(monkeypatch, tmp_path):
    fetches = []
    signer = object()
    monkeypatch.setattr(sampler.chain, "load_arena_cutover", lambda _:None)
    monkeypatch.setattr(sampler, "ArenaStore", lambda _:SimpleNamespace(
        collection_reward_start_epoch=lambda:100,
        get_weight_state=lambda _net,_uid,epoch:{"state_doc":{"valid_from_block":1000+360*epoch}}))
    monkeypatch.setattr(sampler.chain, "connect_substrate", lambda _:None)
    monkeypatch.setattr(sampler.chain, "ArenaChain", lambda *a:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(sampler.signing, "KmsSigner", lambda *a, **kw:signer)
    monkeypatch.setattr(sampler, "HttpsJsonRpcTransport", lambda *a, **kw:SimpleNamespace(close=lambda:None))
    def collect(graph, rpc, db, key, cutover, boundary, epoch, validators):
        fetches.append(epoch)
        return {"epoch":epoch,"boundary":boundary,"result":"burn_only","miner_emissions_rao":{"burn":3}}
    monkeypatch.setattr(sampler,"receipt_at",collect)
    def observe(previous, current):
        return sampler.observe({"LAB_ARENA_SIGNING_KEY_ID":"test"}, object(), tmp_path, previous,
                               {"epoch":current,"last_epoch_block":1000+current*360}, watch.VALIDATORS)
    assert observe({},99) == {"result":"before_retirement"}
    first = observe({},102)
    assert first["epoch"] == 101 and not first["epochs_to_catch_up"]
    assert observe(first,102) == first and fetches==[101]
    second = observe(first,104)
    assert second["epoch"] == 102 and second["epochs_to_catch_up"]==1
    third = observe(second,104)
    assert third["epoch"] == 103 and not third["epochs_to_catch_up"]
    assert fetches==[101,102,103]
    path = tmp_path / "distributions" / "103.json"
    assert json.loads(path.read_text())["miner_emissions_rao"] == {"burn":3}
    path.write_text("{}")
    with pytest.raises(ValueError,match="receipt|differs"):
        observe(third,104)
    with pytest.raises(ValueError,match="changed"):
        observe(second,104)
