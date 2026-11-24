"""Runtime 493 reveal proofs measured from two independent public RPCs."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from leadpoet_canonical.subtensor_events_v2 import (
    SubtensorEventsV2Error, load_subtensor_events_profile_v2,
    prove_timelocked_weights_reveal_v2, validate_subtensor_events_profile_v2,
    supports_timelocked_queue_reveal_v2,
)
from scripts.measure_subtensor_events_profile import profile_from_fixture


ROOT = Path(__file__).resolve().parent / "fixtures"
METADATA_SHA256 = "4ed0c6997e668a995c9dc4ea4c8d587d1625c193fc3361a7781e49156a8db9df"
CODE_HASH = "0x51adea39b16a2b9d60b58d430246d49312f1111b74607edc1166e47a73de4591"
BLOCKS = {
    9549764: ("0xc10a0f91b6d061c2bd593f42f6cca38354364256d1475bf79f760fcd9fbb73e6",
              "d5459f5ae04654cf0b105ab41d62ebe8a00b80e2265e83d9d40d47b25db264f6", 199),
    9549766: ("0x63d10ad2bfc2c2cc930920a1420f06e7f6f03467e6dae721c043330756bcf1ca",
              "d13b01c5734f0ec84532a189c0f6c7ae8df9542d1f87d94ac2231531c9cf4b51", 227),
}


def _measured(block=9549764):
    fixture = json.loads((ROOT / f"subtensor_events_spec493_block{block}.json").read_text())
    profile = load_subtensor_events_profile_v2(spec_version=493)
    metadata = bytes.fromhex(fixture["metadata_hex"][2:])
    events = bytes.fromhex(fixture["system_events"][2:])
    count = bytes.fromhex(fixture["system_event_count"][2:])
    block_hash, events_hash, event_count = BLOCKS[block]
    assert fixture["block_number"] == block
    assert fixture["block_hash"] == block_hash
    assert fixture["metadata_block_number"] == block - 1
    assert fixture["metadata_block_hash"] == "0x" + fixture["parent_hash"]
    assert fixture["runtime_version"]["specVersion"] == 493
    assert fixture["runtime_version"]["transactionVersion"] == 1
    assert len(metadata) == 395608
    assert hashlib.sha256(metadata).hexdigest() == METADATA_SHA256
    assert fixture["runtime_code_storage_hash"] == CODE_HASH
    assert hashlib.sha256(events).hexdigest() == events_hash
    assert int.from_bytes(count, "little") == event_count
    derived = profile_from_fixture(fixture)
    if block == 9549764:
        assert derived == profile
    else:
        assert {k: v for k, v in derived.items() if k != "measurement"} == {
            k: v for k, v in profile.items() if k != "measurement"}
    assert not supports_timelocked_queue_reveal_v2(profile)
    validated = validate_subtensor_events_profile_v2(
        profile, genesis_hash=fixture["genesis_hash"], spec_version=493,
        transaction_version=1, metadata_raw=metadata, runtime_code_hash=CODE_HASH,
    )
    return fixture, validated, events, count


@pytest.mark.parametrize("block", BLOCKS)
def test_spec493_exact_reveals_and_independent_scale_decoder(block):
    fixture, profile, events, count = _measured(block)
    configuration = RuntimeConfigurationObject()
    configuration.update_type_registry(load_type_registry_preset("core"))
    metadata = configuration.create_scale_object(
        "MetadataVersioned", data=ScaleBytes(fixture["metadata_hex"]))
    metadata.decode()
    configuration.add_portable_registry(metadata)
    scale = configuration.create_scale_object(
        "scale_info::" + str(profile["storage"]["events"]["type_id"]),
        data=ScaleBytes(fixture["system_events"]), metadata=metadata,
    )
    records = scale.decode()
    assert scale.data.offset == len(events)
    assert len(records) == int.from_bytes(count, "little")
    for expected in [fixture["expected"], *fixture["other_reveals"]]:
        proof = prove_timelocked_weights_reveal_v2(
            events, profile=profile, event_count_raw=count,
            expected_netuid=expected["netuid"], expected_uid=expected["uid"],
            expected_account_id_hex=expected["account_id_hex"],
        )
        assert proof["uid"] == expected["uid"]
        assert proof["account_id_hex"] == expected["account_id_hex"]
        for index_field, event, attributes in [
            ("weights_set_record_index", "WeightsSet", [71, expected["uid"]]),
            ("reveal_record_index", "TimelockedWeightsRevealed",
             [71, "0x" + expected["account_id_hex"]]),
        ]:
            assert proof[index_field] == expected[index_field]
            record = records[expected[index_field]]
            assert record["phase"] == "Initialization"
            assert record["module_id"] == "SubtensorModule"
            assert record["event_id"] == event
            assert list(record["attributes"]) == attributes
            assert record["topics"] == []


def _compact(value):
    assert value < 1 << 14
    return bytes((value << 2,)) if value < 64 else ((value << 2) | 1).to_bytes(2, "little")


@pytest.mark.parametrize("mutation,match", [
    ("metadata", "observed metadata differs"),
    ("code", "observed runtime code differs"),
    ("spec", "observed spec version differs"),
    ("genesis", "observed genesis hash differs"),
    ("transaction", "observed transaction version differs"),
    ("count", "EventCount"), ("owner", "absent or ambiguous"),
    ("uid", "absent or ambiguous"), ("netuid", "absent or ambiguous"),
    ("trailing", "trailing"), ("phase", "absent or ambiguous"),
    ("duplicate", "absent or ambiguous"),
])
def test_spec493_proof_rejects_changed_binding(mutation, match):
    fixture, profile, events, count = _measured()
    expected = copy.deepcopy(fixture["expected"])
    observed = dict(
        genesis_hash=fixture["genesis_hash"], spec_version=493, transaction_version=1,
        metadata_raw=bytes.fromhex(fixture["metadata_hex"][2:]), runtime_code_hash=CODE_HASH,
    )
    if mutation == "metadata":
        raw = observed["metadata_raw"]
        observed["metadata_raw"] = raw[:-1] + bytes((raw[-1] ^ 1,))
    elif mutation == "code":
        observed["runtime_code_hash"] = "0x" + "00" * 32
    elif mutation == "spec":
        observed["spec_version"] = 491
    elif mutation == "genesis":
        observed["genesis_hash"] = "0x" + "00" * 32
    elif mutation == "transaction":
        observed["transaction_version"] = 2
    elif mutation == "count":
        count = (int.from_bytes(count, "little") + 1).to_bytes(4, "little")
    elif mutation == "owner":
        expected["account_id_hex"] = "00" * 32
    elif mutation in {"uid", "netuid"}:
        expected[mutation] += 1
    elif mutation == "trailing":
        events += b"\x00"
    elif mutation in {"phase", "duplicate"}:
        pair = (b"\x02\x07\x05\x47\x00" + expected["uid"].to_bytes(2, "little")
                + b"\x00\x02\x07\x6d\x47\x00"
                + bytes.fromhex(expected["account_id_hex"]) + b"\x00")
        assert events.count(pair) == 1
        if mutation == "phase":
            events = events.replace(pair, b"\x01" + pair[1:], 1)
        else:
            original_count = int.from_bytes(count, "little")
            events = (_compact(original_count + 2)
                      + events[len(_compact(original_count)):] + pair)
            count = (original_count + 2).to_bytes(4, "little")
    with pytest.raises(SubtensorEventsV2Error, match=match):
        validated = validate_subtensor_events_profile_v2(profile, **observed)
        prove_timelocked_weights_reveal_v2(
            events, profile=validated, event_count_raw=count,
            expected_netuid=expected["netuid"], expected_uid=expected["uid"],
            expected_account_id_hex=expected["account_id_hex"],
        )


@pytest.mark.parametrize("unknown_spec", [489, 495])
def test_unmeasured_runtime_still_fails_closed(unknown_spec):
    with pytest.raises(SubtensorEventsV2Error, match="unavailable"):
        load_subtensor_events_profile_v2(spec_version=unknown_spec)
