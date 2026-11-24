"""Runtime 492 reveal proofs measured from two independent public RPCs."""

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
METADATA_SHA256 = "ca6174da65c56320bc449b0c1c98d6202c65339c40955eebcc914538a27bb878"
CODE_HASH = "0x22786f72f280dc968f38a10627c085eaf3ea530dd3a1621a4776882851ac3d8e"
BLOCKS = {
    9547602: ("0xd88f8dfbd8da2b705e15e980f976243459411ab73d1d8c56a3a5d3e1f3613345",
              "e527a9ff475b4e64799dc3ca2750ad8bf5e96144bde1e8281e47a9954338fece", 222),
    9547606: ("0x3a009a42edcd64706cd9d8270aceba6ca06885ba4ac982465bc15e05ab7537b1",
              "f59afadeb442ffe4c07f68f76ee25ce7a6147768d3e2bedbf99e1132b026688a", 206),
    9547962: ("0x61d8e9f27a8e98889877713552ea6438fb48f58f7ec649ef438064e7d59fb6de",
              "162496a80e8bc6d8ee4fa9ed8ae77ce355c338fcf696f84e8c1e2661b40f3c92", 234),
}


def _measured(block=9547602):
    fixture = json.loads((ROOT / f"subtensor_events_spec492_block{block}.json").read_text())
    profile = load_subtensor_events_profile_v2(spec_version=492)
    metadata = bytes.fromhex(fixture["metadata_hex"][2:])
    events = bytes.fromhex(fixture["system_events"][2:])
    count = bytes.fromhex(fixture["system_event_count"][2:])
    block_hash, events_hash, event_count = BLOCKS[block]
    assert fixture["block_number"] == block
    assert fixture["block_hash"] == block_hash
    assert fixture["metadata_block_number"] == block - 1
    assert fixture["metadata_block_hash"] == "0x" + fixture["parent_hash"]
    assert fixture["runtime_version"]["specVersion"] == 492
    assert fixture["runtime_version"]["transactionVersion"] == 1
    assert len(metadata) == 395745
    assert hashlib.sha256(metadata).hexdigest() == METADATA_SHA256
    assert fixture["runtime_code_storage_hash"] == CODE_HASH
    assert hashlib.sha256(events).hexdigest() == events_hash
    assert int.from_bytes(count, "little") == event_count
    derived = profile_from_fixture(fixture)
    if block == 9547602:
        assert derived == profile
    else:
        assert {k: v for k, v in derived.items() if k != "measurement"} == {
            k: v for k, v in profile.items() if k != "measurement"}
    assert not supports_timelocked_queue_reveal_v2(profile)
    validated = validate_subtensor_events_profile_v2(
        profile, genesis_hash=fixture["genesis_hash"], spec_version=492,
        transaction_version=1, metadata_raw=metadata, runtime_code_hash=CODE_HASH,
    )
    return fixture, validated, events, count


@pytest.mark.parametrize("block", BLOCKS)
def test_spec492_exact_reveals_and_independent_scale_decoder(block):
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
def test_spec492_proof_rejects_changed_binding(mutation, match):
    fixture, profile, events, count = _measured()
    expected = copy.deepcopy(fixture["expected"])
    observed = dict(
        genesis_hash=fixture["genesis_hash"], spec_version=492, transaction_version=1,
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
