"""Pin runtime 491 proofs to independently checked public SN71 reveal bytes."""

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


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/subtensor_events_spec491_block9523485.json"
LATER_FIXTURE = ROOT / "tests/fixtures/subtensor_events_spec491_block9523489.json"
METADATA_SHA256 = "5e7d5b553f5fd70d2fa6e53a529f28c811e8c23686ccae45975bcfeb053df137"
CODE_HASH = "0x4201278656a17aff924ca24123c8e5866d569a84a7150bb0372bc7078fc6173c"


def _measured():
    fixture = json.loads(FIXTURE.read_text())
    profile = load_subtensor_events_profile_v2(spec_version=491)
    metadata = bytes.fromhex(fixture["metadata_hex"][2:])
    events = bytes.fromhex(fixture["system_events"][2:])
    count = bytes.fromhex(fixture["system_event_count"][2:])
    # Both official archive and entrypoint RPCs returned these exact bytes.
    assert fixture["block_number"] == 9523485
    assert fixture["block_hash"] == "0x15a6707111113bbfb4e02660cc63e4a5b4fa844930b58d2d35dba61a625c30bb"
    assert fixture["metadata_block_number"] == fixture["block_number"] - 1
    assert fixture["metadata_block_hash"] == "0x" + fixture["parent_hash"]
    assert fixture["runtime_version"]["specVersion"] == 491
    assert fixture["runtime_version"]["transactionVersion"] == 1
    assert len(metadata) == 395751
    assert hashlib.sha256(metadata).hexdigest() == METADATA_SHA256
    assert fixture["runtime_code_storage_hash"] == CODE_HASH
    assert hashlib.sha256(events).hexdigest() == "1afc4459983d40118792718a2e0bf52c9622ee889fbec7722c4fee1e87f9c455"
    assert hashlib.sha256(count).hexdigest() == "3588f8328fc3447802d7d6cbd1ff995536be64263ebdc03880c3e88ab3a7b238"
    assert int.from_bytes(count, "little") == 215
    assert profile_from_fixture(fixture) == profile
    assert not supports_timelocked_queue_reveal_v2(profile)
    validated = validate_subtensor_events_profile_v2(
        profile, genesis_hash=fixture["genesis_hash"], spec_version=491,
        transaction_version=1, metadata_raw=metadata,
        runtime_code_hash=fixture["runtime_code_storage_hash"],
    )
    return fixture, validated, events, count


@pytest.mark.parametrize("actor", ["expected", "other_reveals"])
def test_spec491_exact_reveal_pair(actor):
    fixture, profile, events, count = _measured()
    expected = (fixture["expected"] if actor == "expected"
                else fixture["other_reveals"][0])
    proof = prove_timelocked_weights_reveal_v2(
        events, profile=profile, event_count_raw=count,
        expected_netuid=expected["netuid"], expected_uid=expected["uid"],
        expected_account_id_hex=expected["account_id_hex"],
    )
    assert (proof["weights_set_record_index"], proof["reveal_record_index"]) == (
        expected["weights_set_record_index"], expected["reveal_record_index"])
    assert proof["uid"] == expected["uid"]
    assert proof["account_id_hex"] == expected["account_id_hex"]


@pytest.mark.parametrize("fixture_path", [FIXTURE, LATER_FIXTURE])
def test_spec491_independent_scale_decoder_confirms_reveals(fixture_path):
    measured, profile, _, _ = _measured()
    fixture = json.loads(fixture_path.read_text())
    events = bytes.fromhex(fixture["system_events"][2:])
    count = bytes.fromhex(fixture["system_event_count"][2:])
    configuration = RuntimeConfigurationObject()
    configuration.update_type_registry(load_type_registry_preset("core"))
    metadata = configuration.create_scale_object(
        "MetadataVersioned", data=ScaleBytes(measured["metadata_hex"])
    )
    metadata.decode()
    configuration.add_portable_registry(metadata)
    scale = configuration.create_scale_object(
        "scale_info::" + str(profile["storage"]["events"]["type_id"]),
        data=ScaleBytes(fixture["system_events"]), metadata=metadata,
    )
    records = scale.decode()
    assert scale.data.offset == len(events)
    assert len(records) == int.from_bytes(count, "little")
    for expected in [fixture["expected"], *fixture.get("other_reveals", [])]:
        for index, event, attributes in [
            (expected["weights_set_record_index"], "WeightsSet", [71, expected["uid"]]),
            (expected["reveal_record_index"], "TimelockedWeightsRevealed",
             [71, "0x" + expected["account_id_hex"]]),
        ]:
            record = records[index]
            assert record["phase"] == "Initialization"
            assert record["module_id"] == "SubtensorModule"
            assert record["event_id"] == event
            assert list(record["attributes"]) == attributes
            assert record["topics"] == []


def test_spec491_later_third_validator_reveal_pair():
    fixture = json.loads(LATER_FIXTURE.read_text())
    profile = load_subtensor_events_profile_v2(spec_version=491)
    assert fixture["block_number"] == 9523489
    assert fixture["block_hash"] == "0xd70b4ee04c808db4b0a3662453674df79694e4342f2be30d8e9a074fcd5e0a8b"
    assert hashlib.sha256(bytes.fromhex(fixture["system_events"][2:])).hexdigest() == "cc19fa959d358be3cfc06f4ec208c297f5a7c5c11d206f90110b86fb3a8e3621"
    assert hashlib.sha256(bytes.fromhex(fixture["system_event_count"][2:])).hexdigest() == "745dde1805da5899278dfdaf99f68076f6d1a42e145e028d39b0975d39089e96"
    assert fixture["metadata_block_hash"] == "0x" + fixture["parent_hash"]
    assert fixture["metadata_raw_sha256"] == METADATA_SHA256
    assert fixture["runtime_code_storage_hash"] == CODE_HASH
    validated = validate_subtensor_events_profile_v2(
        profile, genesis_hash=profile["genesis_hash"], spec_version=491,
        transaction_version=fixture["runtime_version"]["transactionVersion"],
        metadata_sha256=fixture["metadata_raw_sha256"],
        runtime_code_hash=fixture["runtime_code_storage_hash"],
    )
    expected = fixture["expected"]
    proof = prove_timelocked_weights_reveal_v2(
        bytes.fromhex(fixture["system_events"][2:]), profile=validated,
        event_count_raw=bytes.fromhex(fixture["system_event_count"][2:]),
        expected_netuid=expected["netuid"], expected_uid=expected["uid"],
        expected_account_id_hex=expected["account_id_hex"],
    )
    assert (proof["weights_set_record_index"], proof["reveal_record_index"]) == (67, 68)


def _compact(value):
    assert value < 1 << 14
    return bytes((value << 2,)) if value < 64 else ((value << 2) | 1).to_bytes(2, "little")


@pytest.mark.parametrize("mutation,match", [
    ("metadata", "observed metadata differs"),
    ("code", "observed runtime code differs"),
    ("spec", "observed spec version differs"),
    ("genesis", "observed genesis hash differs"),
    ("transaction", "observed transaction version differs"),
    ("count", "EventCount"),
    ("owner", "absent or ambiguous"),
    ("uid", "absent or ambiguous"),
    ("netuid", "absent or ambiguous"),
    ("trailing", "trailing"),
    ("phase", "absent or ambiguous"),
    ("duplicate", "absent or ambiguous"),
])
def test_spec491_proof_rejects_changed_binding(mutation, match):
    fixture, profile, events, count = _measured()
    expected = copy.deepcopy(fixture["expected"])
    observed = dict(
        genesis_hash=fixture["genesis_hash"], spec_version=491, transaction_version=1,
        metadata_raw=bytes.fromhex(fixture["metadata_hex"][2:]), runtime_code_hash=CODE_HASH,
    )
    if mutation == "metadata":
        raw = observed["metadata_raw"]
        observed["metadata_raw"] = raw[:-1] + bytes((raw[-1] ^ 1,))
    elif mutation == "code":
        observed["runtime_code_hash"] = "0x" + "00" * 32
    elif mutation == "spec":
        observed["spec_version"] = 490
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
        # Independently encode the exact two target records, including topics.
        pair = (b"\x02\x07\x05\x47\x00\x00\x00\x00"
                + b"\x02\x07\x6d\x47\x00"
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


@pytest.mark.parametrize("unknown_spec", [489, 496])
def test_unmeasured_runtime_still_fails_closed(unknown_spec):
    with pytest.raises(SubtensorEventsV2Error, match="unavailable"):
        load_subtensor_events_profile_v2(spec_version=unknown_spec)
