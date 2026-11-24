"""Independently decode Finney runtime 494's measured SN71 reveal events."""

import json
from pathlib import Path

import pytest
from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from leadpoet_canonical.subtensor_events_v2 import (
    SubtensorEventsV2Error, load_subtensor_events_profile_v2,
    prove_timelocked_weights_reveal_v2, validate_subtensor_events_profile_v2,
)
from scripts.measure_subtensor_events_profile import profile_from_fixture


@pytest.mark.parametrize("block", [9573884, 9573886])
def test_measured_runtime494_reveal(block):
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          f"subtensor_events_spec494_block{block}.json").read_bytes())
    registered = load_subtensor_events_profile_v2(spec_version=494)
    measured = profile_from_fixture(fixture)
    assert {k: v for k, v in measured.items() if k != "measurement"} == {
        k: v for k, v in registered.items() if k != "measurement"}
    if block == 9573884:
        assert registered == measured
    observed = dict(genesis_hash=fixture["genesis_hash"], spec_version=494,
                    transaction_version=2, metadata_raw=bytes.fromhex(fixture["metadata_hex"][2:]),
                    runtime_code_hash=fixture["runtime_code_storage_hash"])
    validate_subtensor_events_profile_v2(registered, **observed)
    with pytest.raises(SubtensorEventsV2Error, match="transaction version"):
        validate_subtensor_events_profile_v2(registered, **{**observed, "transaction_version": 1})

    config = RuntimeConfigurationObject()
    config.update_type_registry(load_type_registry_preset("core"))
    metadata = config.create_scale_object("MetadataVersioned", data=ScaleBytes(fixture["metadata_hex"]))
    metadata.decode()
    config.add_portable_registry(metadata)
    decoder = config.create_scale_object(
        "scale_info::" + str(measured["storage"]["events"]["type_id"]),
        data=ScaleBytes(fixture["system_events"]), metadata=metadata)
    records = decoder.decode()
    events = bytes.fromhex(fixture["system_events"][2:])
    count = bytes.fromhex(fixture["system_event_count"][2:])
    assert decoder.data.offset == len(events) and len(records) == int.from_bytes(count, "little")
    for owner in [fixture["expected"], *fixture["other_reveals"]]:
        proof = prove_timelocked_weights_reveal_v2(
            events, profile=registered, event_count_raw=count, expected_netuid=71,
            expected_uid=owner["uid"], expected_account_id_hex=owner["account_id_hex"])
        weights = records[proof["weights_set_record_index"]]
        reveal = records[proof["reveal_record_index"]]
        assert weights["event_id"] == "WeightsSet" and list(weights["attributes"]) == [71, owner["uid"]]
        assert reveal["phase"] == weights["phase"] == "Initialization"
        assert reveal["event_id"] == "TimelockedWeightsRevealed"
        assert list(reveal["attributes"]) == [71, "0x" + owner["account_id_hex"]]
