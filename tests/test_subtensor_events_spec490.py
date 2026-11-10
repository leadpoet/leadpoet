"""Pin the spec 490 reveal witnesses to exact public archive bytes."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from leadpoet_canonical.subtensor_events_v2 import (
    SubtensorEventsV2Error, load_subtensor_events_profile_v2,
    prove_timelocked_weights_reveal_v2, validate_subtensor_events_profile_v2,
    supports_timelocked_queue_reveal_v2,
)
from scripts.measure_subtensor_events_profile import profile_from_fixture


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/subtensor_events_spec490_block9478485.json"
LATER_FIXTURE = ROOT / "tests/fixtures/subtensor_events_spec490_block9478489.json"


def test_offline_measurer_recreates_prior_reviewed_profile():
    prior_fixture = json.loads((ROOT / "tests/fixtures/subtensor_events_spec488_block9443929.json").read_text())
    prior_profile = load_subtensor_events_profile_v2(spec_version=488)
    prior_fixture["genesis_hash"] = prior_profile["genesis_hash"]
    assert profile_from_fixture(prior_fixture) == prior_profile


def _measured():
    fixture = json.loads(FIXTURE.read_text())
    profile = load_subtensor_events_profile_v2(spec_version=490)
    metadata = bytes.fromhex(fixture["metadata_hex"][2:])
    events = bytes.fromhex(fixture["system_events"][2:])
    count = bytes.fromhex(fixture["system_event_count"][2:])
    assert fixture["block_number"] == 9478485
    assert fixture["metadata_block_number"] == fixture["block_number"] - 1
    assert fixture["metadata_block_hash"] == "0x" + fixture["parent_hash"]
    assert fixture["runtime_version"]["specVersion"] == 490
    assert fixture["runtime_version"]["transactionVersion"] == 1
    assert hashlib.sha256(metadata).hexdigest() == profile["metadata_raw_sha256"]
    assert profile_from_fixture(fixture) == profile
    assert not supports_timelocked_queue_reveal_v2(profile)
    validated = validate_subtensor_events_profile_v2(
        profile, genesis_hash=fixture["genesis_hash"], spec_version=490,
        transaction_version=1, metadata_raw=metadata,
        runtime_code_hash=fixture["runtime_code_storage_hash"],
    )
    return fixture, validated, events, count


@pytest.mark.parametrize("actor", ["expected", "other_reveals"])
def test_spec490_exact_reveal_pair(actor):
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


def test_spec490_later_third_validator_reveal_pair():
    fixture = json.loads(LATER_FIXTURE.read_text())
    profile = load_subtensor_events_profile_v2(spec_version=490)
    assert fixture["block_number"] == 9478489
    assert fixture["metadata_block_hash"] == "0x" + fixture["parent_hash"]
    assert fixture["metadata_raw_sha256"] == profile["metadata_raw_sha256"]
    validated = validate_subtensor_events_profile_v2(
        profile, genesis_hash=profile["genesis_hash"], spec_version=490,
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


@pytest.mark.parametrize("mutation,match", [
    ("metadata", "observed metadata differs"),
    ("code", "observed runtime code differs"),
    ("spec", "observed spec version differs"),
    ("count", "EventCount"),
    ("owner", "absent or ambiguous"),
    ("uid", "absent or ambiguous"),
    ("trailing", "trailing"),
])
def test_spec490_proof_rejects_changed_binding(mutation, match):
    fixture, profile, events, count = _measured()
    expected = copy.deepcopy(fixture["expected"])
    if mutation == "metadata":
        with pytest.raises(SubtensorEventsV2Error, match=match):
            validate_subtensor_events_profile_v2(
                profile, genesis_hash=fixture["genesis_hash"], spec_version=490,
                transaction_version=1,
                metadata_raw=bytes.fromhex(fixture["metadata_hex"][2:])[:-1] + b"\x00",
                runtime_code_hash=fixture["runtime_code_storage_hash"],
            )
        return
    if mutation == "code":
        with pytest.raises(SubtensorEventsV2Error, match=match):
            validate_subtensor_events_profile_v2(
                profile, genesis_hash=fixture["genesis_hash"], spec_version=490,
                transaction_version=1,
                metadata_raw=bytes.fromhex(fixture["metadata_hex"][2:]),
                runtime_code_hash="0x" + "00" * 32,
            )
        return
    if mutation == "spec":
        with pytest.raises(SubtensorEventsV2Error, match=match):
            validate_subtensor_events_profile_v2(
                profile, genesis_hash=fixture["genesis_hash"], spec_version=488,
                transaction_version=1,
                metadata_raw=bytes.fromhex(fixture["metadata_hex"][2:]),
                runtime_code_hash=fixture["runtime_code_storage_hash"],
            )
        return
    if mutation == "count":
        count = (int.from_bytes(count, "little") + 1).to_bytes(4, "little")
    elif mutation == "owner":
        expected["account_id_hex"] = "00" * 32
    elif mutation == "uid":
        expected["uid"] += 1
    elif mutation == "trailing":
        events += b"\x00"
    with pytest.raises(SubtensorEventsV2Error, match=match):
        prove_timelocked_weights_reveal_v2(
            events, profile=profile, event_count_raw=count,
            expected_netuid=expected["netuid"], expected_uid=expected["uid"],
            expected_account_id_hex=expected["account_id_hex"],
        )


@pytest.mark.parametrize("unknown_spec", [489, 491])
def test_unmeasured_runtime_still_fails_closed(unknown_spec):
    with pytest.raises(SubtensorEventsV2Error, match="unavailable"):
        load_subtensor_events_profile_v2(spec_version=unknown_spec)
