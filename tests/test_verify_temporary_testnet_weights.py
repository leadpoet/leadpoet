from __future__ import annotations

import json

import pytest

from gateway.tee.release_channel_v2 import (
    build_release_channel_v2,
    build_release_lineage_v2,
)
from gateway.tee.release_lineage_v2 import (
    build_compact_release_lineage_boot_verifier_v2,
)
from scripts import verify_temporary_testnet_weights as verifier
from tests.test_release_channel_v2 import (
    _gateway_manifest,
    _validator_manifest,
)


CURRENT_COMMIT = "a" * 40
EXTRA_COMMIT = "b" * 40


def _channel(commit):
    return build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(commit),
        validator_release_manifest=_validator_manifest(commit),
    )


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _approved_lineage(tmp_path, commits=("1" * 40, "2" * 40, "3" * 40, CURRENT_COMMIT)):
    channels = [_channel(commit) for commit in commits]
    runtime_lineage = build_release_lineage_v2(channels, current_commit=CURRENT_COMMIT)
    channel_store = tmp_path / "release-channels-v2.json"
    _write_json(channel_store, {item["commit_sha"]: item for item in channels})
    approved = verifier.build_approved_release_lineage(
        candidate=CURRENT_COMMIT,
        gateway_release=channels[-1]["gateway_release_manifest"],
        validator_release=channels[-1]["validator_release_manifest"],
        runtime_lineage=runtime_lineage,
        release_channels_path=channel_store,
    )
    return channels, approved, channel_store


def test_four_release_lineage_verifies_old_gateway_boot(tmp_path):
    channels, approved, _store = _approved_lineage(tmp_path)
    prior_commit = channels[0]["commit_sha"]
    prior_role = approved["releases"][prior_commit]["roles"]["gateway_scoring"]
    prior_boot = {
        "physical_role": "gateway_scoring",
        **prior_role,
    }
    observed = []
    boot_verifier = build_compact_release_lineage_boot_verifier_v2(
        approved,
        boot_verifier=lambda identity, **kwargs: observed.append(kwargs) or identity,
    )

    assert boot_verifier(prior_boot) == prior_boot
    assert observed == [
        {
            "expected_pcr0": prior_role["pcr0"],
            "certificate_validity_at_attestation_time": True,
        }
    ]


@pytest.mark.parametrize("mutation", ("missing", "foreign", "tampered"))
def test_release_channel_store_must_match_runtime_exactly(tmp_path, mutation):
    channels = [_channel(commit) for commit in ("1" * 40, "2" * 40, "3" * 40, CURRENT_COMMIT)]
    lineage = build_release_lineage_v2(channels, current_commit=CURRENT_COMMIT)
    store = {item["commit_sha"]: item for item in channels}
    if mutation == "missing":
        store.pop(channels[0]["commit_sha"])
    elif mutation == "foreign":
        store[EXTRA_COMMIT] = _channel(EXTRA_COMMIT)
    else:
        store[channels[0]["commit_sha"]]["channel_hash"] = "sha256:" + "0" * 64
    path = tmp_path / "release-channels-v2.json"
    _write_json(path, store)
    with pytest.raises((RuntimeError, ValueError), match="channel|hash"):
        verifier.build_approved_release_lineage(
            candidate=CURRENT_COMMIT,
            gateway_release=channels[-1]["gateway_release_manifest"],
            validator_release=channels[-1]["validator_release_manifest"],
            runtime_lineage=lineage,
            release_channels_path=path,
        )


class _Result:
    def __init__(self, value):
        self.value = value


class _RevealSubstrate:
    def __init__(self, events):
        self.events = events

    @staticmethod
    def get_block_hash(block):
        return "0x" + f"{block:064x}"

    def get_events(self, *, block_hash):
        return list(self.events.get(int(block_hash, 16), ()))


def _reveal_event(attributes=(verifier.NETUID, verifier.EXPECTED_VALIDATOR)):
    return {
        "phase": "Initialization",
        "event": {
            "module_id": "SubtensorModule",
            "event_id": "TimelockedWeightsRevealed",
            "attributes": list(attributes),
        },
    }


def _reveal_query(module, storage, params, *, block_hash):
    assert module == "SubtensorModule"
    block = int(block_hash, 16)
    if storage == "SubnetEpochIndex":
        return _Result(22060 if block == 100 else 22061)
    assert storage == "Weights"
    assert params == [verifier.NETUID, 9]
    return _Result([(0, 65535), (11, 21845)])


def test_reveal_event_is_bound_to_actual_epoch_block_and_vector():
    substrate = _RevealSubstrate({101: [_reveal_event()]})

    result = verifier.prove_finalized_reveal_event(
        substrate,
        query=_reveal_query,
        validator_uid=9,
        commit_block=100,
        target_subnet_epoch_index=22060,
        reveal_period_epochs=1,
        finalized_head_block=103,
        finalized_head_hash=substrate.get_block_hash(103),
        expected_pairs=[(0, 65535), (11, 21845)],
    )

    assert result == {
        "reveal_event": "SubtensorModule.TimelockedWeightsRevealed",
        "reveal_event_block": 101,
        "reveal_event_block_hash": substrate.get_block_hash(101),
        "reveal_event_record_index": 0,
        "reveal_event_subnet_epoch_index": 22061,
    }


def test_unchanged_old_vector_does_not_replace_missing_reveal_event():
    substrate = _RevealSubstrate({})

    with pytest.raises(RuntimeError, match="reveal event is absent"):
        verifier.prove_finalized_reveal_event(
            substrate,
            query=_reveal_query,
            validator_uid=9,
            commit_block=100,
            target_subnet_epoch_index=22060,
            reveal_period_epochs=1,
            finalized_head_block=103,
            finalized_head_hash=substrate.get_block_hash(103),
            expected_pairs=[(0, 65535), (11, 21845)],
        )


def test_reveal_event_is_unique_and_its_block_vector_must_match():
    duplicate = _RevealSubstrate(
        {101: [_reveal_event()], 102: [_reveal_event()]}
    )
    kwargs = {
        "query": _reveal_query,
        "validator_uid": 9,
        "commit_block": 100,
        "target_subnet_epoch_index": 22060,
        "reveal_period_epochs": 1,
        "finalized_head_block": 103,
        "finalized_head_hash": duplicate.get_block_hash(103),
        "expected_pairs": [(0, 65535), (11, 21845)],
    }
    with pytest.raises(RuntimeError, match="reveal event is absent or ambiguous"):
        verifier.prove_finalized_reveal_event(duplicate, **kwargs)

    single = _RevealSubstrate({101: [_reveal_event()]})
    kwargs["finalized_head_hash"] = single.get_block_hash(103)
    kwargs["expected_pairs"] = [(0, 65535), (11, 21844)]
    with pytest.raises(RuntimeError, match="event block vector differs"):
        verifier.prove_finalized_reveal_event(single, **kwargs)
