from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from leadpoet_canonical.subtensor_events_v2 import (
    PROOF_SCHEMA_VERSION,
    SYSTEM_EVENT_COUNT_STORAGE_KEY,
    SYSTEM_EVENTS_STORAGE_KEY,
    SubtensorEventsV2Error,
    decode_system_events_v2,
    load_subtensor_events_profile_v2,
    prove_timelocked_weights_reveal_v2,
    validate_subtensor_events_profile_v2,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    ROOT / "tests" / "fixtures" / "subtensor_events_spec455_block9039648.json"
)


def _fixture():
    value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return (
        value,
        bytes.fromhex(value["system_events"][2:]),
        bytes.fromhex(value["system_event_count"][2:]),
    )


def _proof(profile, fixture, events_raw, event_count_raw):
    expected = fixture["expected"]
    return prove_timelocked_weights_reveal_v2(
        events_raw,
        profile=profile,
        event_count_raw=event_count_raw,
        expected_netuid=expected["netuid"],
        expected_uid=expected["uid"],
        expected_account_id_hex=expected["account_id_hex"],
    )


def _compact(value):
    if value < 1 << 6:
        return bytes((value << 2,))
    if value < 1 << 14:
        return ((value << 2) | 1).to_bytes(2, "little")
    return ((value << 2) | 2).to_bytes(4, "little")


def _measured_pair_bytes(fixture):
    expected = fixture["expected"]
    return bytes.fromhex(
        "020705"
        + int(expected["netuid"]).to_bytes(2, "little").hex()
        + int(expected["uid"]).to_bytes(2, "little").hex()
        + "00"
        + "02076d"
        + int(expected["netuid"]).to_bytes(2, "little").hex()
        + expected["account_id_hex"]
        + "00"
    )


def test_real_spec455_archive_events_prove_exact_adjacent_reveal():
    profile = load_subtensor_events_profile_v2()
    fixture, events_raw, event_count_raw = _fixture()

    assert profile["spec_version"] == 455
    assert profile["transaction_version"] == 1
    assert profile["metadata_raw_sha256"] == (
        "74c4067de4bf2eba95156e8a46c793b52fcd9862dfeb28502632e46416979ec7"
    )
    assert profile["runtime_code_storage_hash"] == (
        "0x329a9e79cfcd553b8151e65ac474696ee03b36057605a71bc4c9acc811025567"
    )
    assert profile["storage"]["events"]["key"] == SYSTEM_EVENTS_STORAGE_KEY
    assert profile["storage"]["event_count"]["key"] == SYSTEM_EVENT_COUNT_STORAGE_KEY
    assert len(profile["types"]) == 101
    assert fixture["block_hash"] == profile["measurement"]["block_hash"]
    assert len(events_raw) == profile["measurement"]["system_events_bytes"]
    assert (
        hashlib.sha256(events_raw).hexdigest()
        == profile["measurement"]["system_events_sha256"]
    )
    assert (
        hashlib.sha256(event_count_raw).hexdigest()
        == profile["measurement"]["system_event_count_raw_sha256"]
    )

    validated = validate_subtensor_events_profile_v2(
        profile,
        genesis_hash=profile["genesis_hash"],
        spec_version=455,
        transaction_version=1,
        metadata_sha256=profile["metadata_raw_sha256"],
        runtime_code_hash=profile["runtime_code_storage_hash"],
    )
    records = decode_system_events_v2(
        events_raw, profile=validated, event_count_raw=event_count_raw
    )
    assert len(records) == 304
    assert records[55]["phase"] == "Initialization"
    assert records[55]["runtime_event"] == "SubtensorModule"
    assert records[55]["pallet_event"] == "WeightsSet"
    assert records[55]["fields"] == [71, 0]
    assert records[56]["pallet_event"] == "TimelockedWeightsRevealed"
    assert records[56]["fields"] == [
        71,
        "0x" + fixture["expected"]["account_id_hex"],
    ]

    proof = _proof(validated, fixture, events_raw, event_count_raw)
    assert proof == {
        "schema_version": PROOF_SCHEMA_VERSION,
        "profile_sha256": "sha256:026cde124b9061ed80a924466b22c45f1578a23222d47bcf59caff7aa958c13c",
        "events_sha256": "sha256:1bcd34eff64b0499881ea19b59005825dcd7979380c6cd45bf4941b4d8985430",
        "event_count": 304,
        "weights_set_record_index": 55,
        "weights_set_record_sha256": "sha256:55041fe8cf2b7b6e2e2bedafc0c8b8e9cc14d18b61ac1d6822e805f9fed360af",
        "reveal_record_index": 56,
        "reveal_record_sha256": "sha256:1b686f6b4e3a8d7b1feab391fc649780cc07d435e37276adda742fafd0c5f8a3",
        "netuid": 71,
        "uid": 0,
        "account_id_hex": fixture["expected"]["account_id_hex"],
        "phase": "Initialization",
        "runtime_event_index": 7,
        "weights_set_event_index": 5,
        "timelocked_weights_revealed_event_index": 109,
    }


@pytest.mark.parametrize(
    "changed",
    [
        {"expected_netuid": 72},
        {"expected_uid": 24},
        {"expected_account_id_hex": "11" * 32},
    ],
)
def test_wrong_reveal_identity_fails_closed(changed):
    profile = load_subtensor_events_profile_v2()
    fixture, events_raw, event_count_raw = _fixture()
    expected = fixture["expected"]
    arguments = {
        "expected_netuid": expected["netuid"],
        "expected_uid": expected["uid"],
        "expected_account_id_hex": expected["account_id_hex"],
    }
    arguments.update(changed)
    with pytest.raises(SubtensorEventsV2Error, match="absent or ambiguous"):
        prove_timelocked_weights_reveal_v2(
            events_raw,
            profile=profile,
            event_count_raw=event_count_raw,
            **arguments,
        )


def test_duplicate_matching_pair_is_ambiguous_not_success():
    profile = load_subtensor_events_profile_v2()
    fixture, events_raw, _event_count_raw = _fixture()
    pair = _measured_pair_bytes(fixture)
    assert pair in events_raw
    duplicated = _compact(306) + events_raw[2:] + pair
    with pytest.raises(SubtensorEventsV2Error, match="absent or ambiguous"):
        _proof(profile, fixture, duplicated, (306).to_bytes(4, "little"))


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda raw: raw + b"\x00", "trailing bytes"),
        (lambda raw: raw[:-1], "truncated"),
        (
            lambda raw: bytes((0x12, 0x03, 0x00, 0x00)) + raw[2:],
            "not canonical",
        ),
        (
            lambda raw: bytes(raw[:3]) + b"\xff" + bytes(raw[4:]),
            "variant index is unknown",
        ),
    ],
)
def test_malformed_unknown_and_excess_event_bytes_fail_closed(mutator, match):
    profile = load_subtensor_events_profile_v2()
    _fixture_value, events_raw, event_count_raw = _fixture()
    with pytest.raises(SubtensorEventsV2Error, match=match):
        decode_system_events_v2(
            mutator(events_raw), profile=profile, event_count_raw=event_count_raw
        )


def test_event_count_and_collection_bounds_fail_closed():
    profile = load_subtensor_events_profile_v2()
    _fixture_value, events_raw, event_count_raw = _fixture()
    with pytest.raises(SubtensorEventsV2Error, match="differs"):
        decode_system_events_v2(
            events_raw,
            profile=profile,
            event_count_raw=(197).to_bytes(4, "little"),
        )
    with pytest.raises(SubtensorEventsV2Error, match="bytes are invalid"):
        decode_system_events_v2(
            events_raw, profile=profile, event_count_raw=event_count_raw + b"\x00"
        )
    with pytest.raises(SubtensorEventsV2Error, match="exceeds"):
        decode_system_events_v2(
            _compact(15_001),
            profile=profile,
            event_count_raw=(15_001).to_bytes(4, "little"),
        )


def test_topics_are_decoded_and_reveal_bytes_inside_a_topic_are_not_an_event():
    profile = load_subtensor_events_profile_v2()
    fixture, _events_raw, _event_count_raw = _fixture()
    pair = _measured_pair_bytes(fixture)
    weights_without_empty_topics = pair[:7]
    false_reveal_topic = pair[8:40].ljust(32, b"\x00")
    one_record = b"\x04" + weights_without_empty_topics + b"\x04" + false_reveal_topic
    records = decode_system_events_v2(
        one_record, profile=profile, event_count_raw=(1).to_bytes(4, "little")
    )
    assert records[0]["topics"] == ["0x" + false_reveal_topic.hex()]
    with pytest.raises(SubtensorEventsV2Error, match="absent or ambiguous"):
        _proof(profile, fixture, one_record, (1).to_bytes(4, "little"))


def test_runtime_and_metadata_bindings_fail_closed():
    profile = load_subtensor_events_profile_v2()
    common = {
        "genesis_hash": profile["genesis_hash"],
        "spec_version": profile["spec_version"],
        "transaction_version": profile["transaction_version"],
        "metadata_sha256": profile["metadata_raw_sha256"],
        "runtime_code_hash": profile["runtime_code_storage_hash"],
    }
    changes = (
        {"genesis_hash": "00" * 32},
        {"spec_version": 451},
        {"transaction_version": 2},
        {"metadata_sha256": "00" * 32},
        {"runtime_code_hash": "00" * 32},
    )
    for change in changes:
        arguments = dict(common)
        arguments.update(change)
        with pytest.raises(SubtensorEventsV2Error, match="differs"):
            validate_subtensor_events_profile_v2(profile, **arguments)
    with pytest.raises(SubtensorEventsV2Error, match="not SCALE metadata V14"):
        validate_subtensor_events_profile_v2(
            profile,
            genesis_hash=profile["genesis_hash"],
            spec_version=profile["spec_version"],
            transaction_version=profile["transaction_version"],
            metadata_raw=b"meta\x0fwrong",
            runtime_code_hash=profile["runtime_code_storage_hash"],
        )


def test_profile_tampering_and_unreachable_types_fail_closed():
    profile = load_subtensor_events_profile_v2()
    changed_index = copy.deepcopy(profile)
    changed_index["event_layout"]["timelocked_weights_revealed_event_index"] = 110
    with pytest.raises(SubtensorEventsV2Error, match="variant"):
        decode_system_events_v2(
            b"\x00", profile=changed_index, event_count_raw=b"\x00" * 4
        )

    unreachable = copy.deepcopy(profile)
    unreachable["types"]["999"] = {
        "path": [],
        "def": {"primitive": "u8"},
    }
    with pytest.raises(SubtensorEventsV2Error, match="unreachable"):
        decode_system_events_v2(
            b"\x00", profile=unreachable, event_count_raw=b"\x00" * 4
        )

    extra_field = copy.deepcopy(profile)
    extra_field["unmeasured"] = True
    with pytest.raises(SubtensorEventsV2Error, match="fields are invalid"):
        decode_system_events_v2(
            b"\x00", profile=extra_field, event_count_raw=b"\x00" * 4
        )
