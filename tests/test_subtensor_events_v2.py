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
    ROOT / "tests" / "fixtures" / "subtensor_events_spec452_block8984916.json"
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


def test_real_spec452_archive_events_prove_exact_adjacent_reveal():
    profile = load_subtensor_events_profile_v2()
    fixture, events_raw, event_count_raw = _fixture()

    assert profile["spec_version"] == 452
    assert profile["transaction_version"] == 1
    assert profile["metadata_raw_sha256"] == (
        "79fc9235a87651a0cd5b93856d4b5696ffb8a0bd26c6f30a1f1402ac8aaad195"
    )
    assert profile["runtime_code_storage_hash"] == (
        "0x40a8c3c99a47d6739b086236308535fab26d5fd4cc5c88eb83f6a3c8b928f7cc"
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
        spec_version=452,
        transaction_version=1,
        metadata_sha256=profile["metadata_raw_sha256"],
        runtime_code_hash=profile["runtime_code_storage_hash"],
    )
    records = decode_system_events_v2(
        events_raw, profile=validated, event_count_raw=event_count_raw
    )
    assert len(records) == 196
    assert records[1]["phase"] == "Initialization"
    assert records[1]["runtime_event"] == "SubtensorModule"
    assert records[1]["pallet_event"] == "WeightsSet"
    assert records[1]["fields"] == [71, 23]
    assert records[2]["pallet_event"] == "TimelockedWeightsRevealed"
    assert records[2]["fields"] == [
        71,
        "0x" + fixture["expected"]["account_id_hex"],
    ]

    proof = _proof(validated, fixture, events_raw, event_count_raw)
    assert proof == {
        "schema_version": PROOF_SCHEMA_VERSION,
        "profile_sha256": "sha256:fb0520a776397baad431a13cddf8ab093e350757c349c0d6b052a70d0faac4ec",
        "events_sha256": "sha256:eaf06128da2f1bdf48b98209ba43bdf2c7f37b1a9398b7164f8b63387cb0e27a",
        "event_count": 196,
        "weights_set_record_index": 1,
        "weights_set_record_sha256": "sha256:d4d2dfdc11a036a14a7f726822bf5dbb82f13c50ae4d4f794c919d814288979d",
        "reveal_record_index": 2,
        "reveal_record_sha256": "sha256:ea4d2d7ef3d181d12b54ef0fb0db4a571b46f30086fea2f8b3a5b7aeca6f871f",
        "netuid": 71,
        "uid": 23,
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
    duplicated = _compact(198) + events_raw[2:] + pair
    with pytest.raises(SubtensorEventsV2Error, match="absent or ambiguous"):
        _proof(profile, fixture, duplicated, (198).to_bytes(4, "little"))


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
