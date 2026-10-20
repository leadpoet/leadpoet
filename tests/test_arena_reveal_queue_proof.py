"""Exact public v468 history and failure controls through production proof code."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_arena.chain_source import ValidatorChainSourceV2
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_RPC_METHOD, _compact_decode, _compact_encode,
    decode_selective_metagraph_result, decode_subnet_epoch_storage,
    decode_timelocked_weight_commits, decode_reveal_period_epochs_storage,
    encode_selective_metagraph_params, reveal_period_epochs_storage_key,
    subnet_epoch_storage_key, system_events_storage_key, system_event_count_storage_key,
    timelocked_weight_commits_storage_key,
)
from leadpoet_canonical.subtensor_events_v2 import (
    SubtensorEventsV2Error, load_subtensor_events_profile_v2,
    prove_timelocked_weights_queue_reveal_v2, prove_timelocked_weights_reveal_v2,
    validate_subtensor_events_profile_v2,
)

FIXTURE = Path(__file__).parent / "fixtures" / "arena_reveal_queue_spec468_block9118128.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def responses(data):
    return {json.dumps([x["method"], x["params"]], sort_keys=True): x["result"]
            for x in data["rpc_requests"]}


def lookup(data, method, params):
    return copy.deepcopy(responses(data)[json.dumps([method, params], sort_keys=True)])


def production_source(data):
    def rpc(*, method, params, request_id):
        return lookup(data, method, params)
    return ValidatorChainSourceV2(rpc_call=rpc, archive_rpc_call=rpc,
        finalization_sleep=lambda _: None, epoch_authority_supplier=lambda: None)


def production_proof(data):
    return production_source(data).prove_timelocked_reveal_transition(**data["expected"])


def queue_args(data):
    expected = data["expected"]
    proof = data["proof"]
    parent = proof["event_witness"]["queue_transition"]["parent_block_hash"]
    transition = "0x" + proof["reveal_block_hash"]
    profile = load_subtensor_events_profile_v2(spec_version=468)
    profile = validate_subtensor_events_profile_v2(profile,
        genesis_hash=expected["chain_profile"]["genesis_hash"], spec_version=468,
        transaction_version=1,
        metadata_raw=bytes.fromhex(lookup(data, "state_getMetadata", [parent])[2:]),
        runtime_code_hash=lookup(data, "state_getStorageHash", ["0x3a636f6465", parent]))
    storage = lambda key, digest: lookup(data, "state_getStorage", [key, digest])
    key = timelocked_weight_commits_storage_key(netuid=71, subnet_epoch_index=25262)
    graph = lambda digest: decode_selective_metagraph_result(lookup(data, "state_call", [
        CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=71), digest]))
    return dict(events_raw=bytes.fromhex(storage(system_events_storage_key(), transition)[2:]),
        event_count_raw=bytes.fromhex(storage(system_event_count_storage_key(), transition)[2:]),
        profile=profile, expected_netuid=71, expected_uid=proof["validator_uid"],
        expected_account_id_hex=expected["hotkey_public_key_hex"],
        expected_commitment_hex=expected["commitment_hex"],
        expected_reveal_round=expected["reveal_round"], expected_submitted_at=expected["inclusion_block"],
        subnet_epoch_index=expected["subnet_epoch_index"],
        queue_before_raw=storage(key, parent), queue_after_raw=storage(key, transition),
        parent_block_hash=parent, transition_block_hash=transition,
        parent_header=lookup(data, "chain_getHeader", [parent]),
        transition_header=lookup(data, "chain_getHeader", [transition]),
        parent_metagraph=graph(parent), transition_metagraph=graph(transition),
        scheduler={name: decode_subnet_epoch_storage(storage(subnet_epoch_storage_key(
            storage_name=name, netuid=71), parent), storage_name=name) for name in
            ("SubnetEpochIndex", "LastEpochBlock", "Tempo", "PendingEpochAt", "BlocksSinceLastStep")},
        reveal_period_override=decode_reveal_period_epochs_storage(storage(
            reveal_period_epochs_storage_key(netuid=71), parent)))


def encode_queue(entries):
    return "0x" + (_compact_encode(len(entries)) + b"".join(
        bytes.fromhex(x["hotkey_public_key"]) + x["submitted_at"].to_bytes(8, "little")
        + _compact_encode(len(bytes.fromhex(x["commitment_hex"])))
        + bytes.fromhex(x["commitment_hex"]) + x["reveal_round"].to_bytes(8, "little")
        for x in entries)).hex()


def pair(account, uid, *, phase=b"\x02"):
    return (phase + bytes.fromhex("07054700") + uid.to_bytes(2, "little") + b"\x00"
            + phase + bytes.fromhex("076d4700") + bytes.fromhex(account) + b"\x00")


def mutate_events(args, control):
    raw = args["events_raw"]
    count, offset = _compact_decode(raw, 0)
    body = raw[offset:]
    target = pair(args["expected_account_id_hex"], args["expected_uid"])
    assert body.count(target) == 2
    if control == "missing_pair":
        body = body.replace(target, b"", 1); count -= 2
    elif control == "zero_pairs":
        body = body.replace(target, b""); count -= 4
    elif control == "extra_pair":
        body = body.replace(target, target + target, 1); count += 2
    elif control == "wrong_phase":
        body = body.replace(target, pair(args["expected_account_id_hex"], args["expected_uid"],
            phase=b"\x00\x00\x00\x00\x00"), 1)
    elif control == "wrong_account":
        body = body.replace(target, pair("77" * 32, args["expected_uid"]), 1)
    elif control == "wrong_uid":
        body = body.replace(target, pair(args["expected_account_id_hex"], args["expected_uid"] + 1), 1)
    else:
        raise AssertionError(control)
    args["events_raw"] = _compact_encode(count) + body
    args["event_count_raw"] = count.to_bytes(4, "little")


def test_exact_historical_source_proof_and_normal_unique_witness():
    data = fixture()
    result = production_proof(data)
    assert json.loads(json.dumps(result)) == data["proof"]
    queue = result["event_witness"]["queue_transition"]
    assert queue["runtime_policy_source_commit"] == data["provenance"]["runtime_ordering"]["source_commit"]
    assert {r["method"] for r in data["rpc_requests"]} <= {
        "chain_getFinalizedHead", "chain_getBlockHash", "chain_getHeader", "state_call",
        "state_getStorage", "state_getMetadata", "state_getRuntimeVersion", "state_getStorageHash"}
    assert queue["queue_count"] == 3 and queue["selected_queue_index"] == 2
    assert result["event_witness"]["weights_set_record_index"] == 75
    assert result["event_witness"]["reveal_record_index"] == 76
    assert queue["queue_after_storage_present"] is False
    assert queue["queue_after_sha256"] is None
    args = queue_args(data)
    unique = prove_timelocked_weights_reveal_v2(args["events_raw"], profile=args["profile"],
        event_count_raw=args["event_count_raw"], expected_netuid=71, expected_uid=0,
        expected_account_id_hex="924620afb270acb1ee27bd034aa9e97108ef276da5079db982883cd70294741a")
    assert unique["weights_set_record_index"] == 71
    assert "queue_transition" not in unique


def test_actual_primary_unique_control_uses_unchanged_source_path():
    data = fixture()
    result = production_source(data).prove_timelocked_reveal_transition(**data["unique_control_expected"])
    assert json.loads(json.dumps(result)) == data["unique_control_proof"]
    assert result["validator_uid"] == 0
    assert "queue_transition" not in result["event_witness"]


@pytest.mark.parametrize("control", ["missing_pair", "zero_pairs", "extra_pair", "wrong_phase", "wrong_account", "wrong_uid"])
def test_full_source_cannot_escape_multi_entry_guard_through_unique_event(control):
    data = fixture(); args = queue_args(data); mutate_events(args, control)
    transition = args["transition_block_hash"]
    for request in data["rpc_requests"]:
        if request["method"] != "state_getStorage" or request["params"][-1] != transition:
            continue
        if request["params"][0] == system_events_storage_key():
            request["result"] = "0x" + args["events_raw"].hex()
        elif request["params"][0] == system_event_count_storage_key():
            request["result"] = "0x" + args["event_count_raw"].hex()
    with pytest.raises(SubtensorEventsV2Error):
        production_proof(data)


@pytest.mark.parametrize("control", [
    "earlier_target", "wrong_submitted", "wrong_round", "wrong_commitment", "partial_queue",
    "duplicate_target", "reordered_queue", "reordered_accounts", "expired_cleanup", "other_queue", "unreviewed_runtime",
    "tempo_zero", "safety_fire", "pending_fire", "automatic_fire", "parent_hash", "transition_hash",
    "header_digest", "header_state_root", "parent_uid", "transition_uid", "metagraph_block",
    "other_subnet", "replayed_headers", "wrong_period", "default_period_mutation",
])
def test_production_queue_verifier_rejects_authority_mutations(control):
    args = queue_args(fixture())
    before = list(decode_timelocked_weight_commits(args["queue_before_raw"]))
    if control == "earlier_target":
        target = before[1]
        args.update(expected_commitment_hex=target["commitment_hex"],
                    expected_submitted_at=target["submitted_at"], expected_reveal_round=target["reveal_round"])
    elif control == "wrong_submitted": args["expected_submitted_at"] += 1
    elif control == "wrong_round": args["expected_reveal_round"] += 1
    elif control == "wrong_commitment": args["expected_commitment_hex"] = "ab" * 291
    elif control == "partial_queue": args["queue_after_raw"] = encode_queue(before[:1])
    elif control == "duplicate_target": args["queue_before_raw"] = encode_queue(before + before[-1:])
    elif control == "reordered_queue": args["queue_before_raw"] = encode_queue([before[0], before[2], before[1]])
    elif control == "reordered_accounts": args["queue_before_raw"] = encode_queue([before[1], before[0], before[2]])
    elif control == "expired_cleanup": args["scheduler"]["SubnetEpochIndex"] += 1
    elif control == "other_queue": args["subnet_epoch_index"] += 1
    elif control == "unreviewed_runtime": args["profile"]["runtime_code_storage_hash"] = "0x" + "22" * 32
    elif control == "tempo_zero": args["scheduler"]["Tempo"] = 0
    elif control == "safety_fire": args["scheduler"]["BlocksSinceLastStep"] = args["scheduler"]["Tempo"] + 1
    elif control == "pending_fire": args["scheduler"]["PendingEpochAt"] = 9118128
    elif control == "automatic_fire": args["scheduler"]["LastEpochBlock"] = 9118128 - args["scheduler"]["Tempo"]
    elif control == "parent_hash": args["parent_block_hash"] = "0x" + "22" * 32
    elif control == "transition_hash": args["transition_block_hash"] = "0x" + "22" * 32
    elif control == "header_digest": args["transition_header"]["digest"]["logs"].append("0x00")
    elif control == "header_state_root": args["parent_header"]["stateRoot"] = "0x" + "22" * 32
    elif control in ("parent_uid", "transition_uid"):
        key = "parent_metagraph" if control == "parent_uid" else "transition_metagraph"
        args[key]["hotkeys"][155] = args[key]["hotkeys"][154]
    elif control == "metagraph_block": args["parent_metagraph"]["block"] += 1
    elif control == "other_subnet": args["expected_netuid"] += 1
    elif control == "replayed_headers": args["parent_header"] = copy.deepcopy(args["transition_header"])
    elif control == "wrong_period": args["reveal_period_override"] = 2
    elif control == "default_period_mutation": args["profile"]["metadata_raw_sha256"] = "22" * 32
    with pytest.raises(ValueError):
        prove_timelocked_weights_queue_reveal_v2(**args)


def test_absent_and_encoded_empty_storage_are_distinct_proofs():
    args = queue_args(fixture())
    absent = prove_timelocked_weights_queue_reveal_v2(**args)
    args["queue_after_raw"] = "0x00"
    encoded_empty = prove_timelocked_weights_queue_reveal_v2(**args)
    assert absent["queue_transition"] != encoded_empty["queue_transition"]
    assert encoded_empty["queue_transition"]["queue_after_storage_present"] is True


def test_unreviewed_runtime_cannot_enter_duplicate_queue_proof():
    data = fixture()
    for request in data["rpc_requests"]:
        if request["method"] == "state_getRuntimeVersion":
            request["result"]["specVersion"] = 469
    with pytest.raises(SubtensorEventsV2Error):
        production_proof(data)


def test_full_source_recomputed_hash_rejects_replayed_storage_witness():
    data = fixture()
    original = production_proof(data)
    args = queue_args(data)
    key = timelocked_weight_commits_storage_key(netuid=71, subnet_epoch_index=25262)
    for request in data["rpc_requests"]:
        if request["method"] == "state_getStorage" and request["params"] == [key, args["transition_block_hash"]]:
            request["result"] = "0x00"
    changed = production_proof(data)
    # Both encodings mean an empty queue, but a proof for one storage value
    # cannot be replayed for the other. Presence and raw bytes are committed.
    assert changed["transition_hash"] != original["transition_hash"]
