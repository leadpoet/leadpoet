"""Gateway issuance proves frozen models and promotion epochs before signing."""

from copy import deepcopy
from datetime import date, datetime, timezone
from fractions import Fraction
from types import SimpleNamespace

import pytest

from lab_arena import rewards, signing, submission_similarity, weight_state, wiring
from Leadpoet.utils.subnet_epoch import SubnetEpochCutover, SubnetEpochSnapshot
from lab_arena.service import ArenaService, ServiceError
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.slot_decay_kernel_test import ALICE, BOB, BURN, HOTKEYS, _basis, _win
from tests.lab_arena.submission_similarity_runtime_test import archive


EPOCH = 30000
SOURCE_EPOCH = 29300
BLOCK = 10000
WEEK = 140


@pytest.fixture(autouse=True)
def historical_weekly_policy(monkeypatch):
    # These existing tests isolate weekly tenure, using their original signed
    # policy even though their synthetic epoch numbers exceed later cutovers.
    policy = rewards.reward_slot_policy_document(date(2026, 11, 1))
    monkeypatch.setattr(rewards, "reward_slot_policy_document", lambda *_, **__: deepcopy(policy))


def _harness(*, enabled=True):
    signer = signing.LocalSigner.generate()
    winner = _win(delta=20, start=SOURCE_EPOCH, name="winner")
    policy = rewards.reward_slot_policy_document(date(2026, 11, 1))
    first = signing.sign_document(signer, _basis(
        [deepcopy(winner) for _ in range(3)], epoch=SOURCE_EPOCH,
        policy=policy, round_id=winner["round_id"],
    ), hash_field="reward_basis_hash")
    current = signing.sign_document(signer, _basis(
        [deepcopy(winner) for _ in range(3)], epoch=29900, policy=policy,
    ), hash_field="reward_basis_hash")
    payload = archive("def run(icp):\n    return icp.industry\n")
    identity = submission_similarity.inspect_archive(payload)
    submission = {
        "round_id": winner["round_id"], "submission_id": winner["submission_id"],
        "miner_hotkey": winner["miner_hotkey"], "status": "frozen", "is_king": False,
        "source_ref": "frozen-source", "source_size_bytes": len(payload),
        "source_archive_sha256": identity.archive_sha256,
        "source_normalized_sha256": identity.normalized_sha256,
    }
    def row(basis):
        return {
            "round_id": basis["round_id"], "effective_reward_epoch": basis["effective_reward_epoch"],
            "arena_network_name": "finney", "arena_netuid": 71,
            "reward_activated_at": "2026-11-01T00:00:00Z", "rewards_enabled": True,
            "configuration_doc": {"mode": "live", "baseline_hotkey": BURN},
            "reward_basis_doc": basis,
        }
    data = SimpleNamespace(
        current=current, first=first, history=[row(current), row(first)],
        previous=None, existing=None, saved=None, submission=submission, payload=payload,
        history_calls=[], object_calls=[], submission_calls=[], publish_calls=[],
    )
    def history(**kwargs):
        data.history_calls.append(kwargs)
        return deepcopy(data.history)
    def get_submission(submission_id):
        data.submission_calls.append(submission_id)
        return deepcopy(data.submission)
    def get_object(ref, maximum):
        data.object_calls.append((ref, maximum))
        return data.payload
    def publish(network, netuid, epoch, digest, state):
        data.publish_calls.append((network, netuid, epoch, digest))
        data.saved = deepcopy(state)
        return {"state": state}
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        mode="live", weekly_slot_decay_enabled=enabled, accepted_burn_hotkey=BURN,
        chain=SimpleNamespace(accepted_weight_epoch_scope=lambda: {
            "epoch": EPOCH, "genesis_hash": "1" * 64,
            "valid_from_block": BLOCK, "valid_until_block": BLOCK + 100,
        }),
    )
    service._store = SimpleNamespace(
        get_weight_state=lambda _network, _netuid, requested: (
            data.existing if requested == EPOCH else None),
        latest_weight_state=lambda *_args: data.previous,
        published_reward_bases=history, get_submission=get_submission,
        publish_weight_state=publish,
    )
    service._objects = SimpleNamespace(get_bounded=get_object)
    service._chain_scope = lambda: ("finney", 71)
    service._reward_signer = lambda: signer
    service.public_reward_basis = lambda *_args, **_kwargs: data.current
    return service, data, signer


def _previous(data, signer, *, clock=True, epoch=EPOCH - 1, genesis="1" * 64, network="finney", netuid=71):
    kwargs = {}
    if clock:
        kwargs["weekly_slot_decay"] = {
            "epochs_per_halving": WEEK,
            "slots": [{"start_epoch": SOURCE_EPOCH,
                       "source_normalized_sha256": data.submission["source_normalized_sha256"]}
                      for _ in range(3)],
        }
    state = weight_state.build_accepted_weight_state(
        signer, network=network, genesis_hash=genesis, netuid=netuid, epoch=epoch,
        valid_from_block=BLOCK - 10, valid_until_block=BLOCK,
        reward_basis=data.current, burn_hotkey=BURN, issued_at=data.current["published_at"],
        **kwargs,
    )
    data.previous = {key: state[key] for key in ("network", "netuid", "epoch", "state_hash")}
    data.previous["state_doc"] = state
    return state


def _actual_adapter(monkeypatch):
    snapshot = SubnetEpochSnapshot(
        network_genesis_hash="0x" + "1" * 64, netuid=71, head_kind="finalized",
        block_hash="0x" + "2" * 64, current_block=BLOCK + 4,
        last_epoch_block=BLOCK, pending_epoch_at=0, subnet_epoch_index=20,
        tempo=360, blocks_since_last_step=4, observed_at="2026-11-09T00:00:00Z",
    )
    adapter = wiring.ChainReadsAdapter(object())
    adapter._cutover = SubnetEpochCutover(
        network_genesis_hash=snapshot.network_genesis_hash, netuid=71,
        cutover_block=BLOCK, cutover_block_hash="0x" + "3" * 64,
        first_subnet_epoch_index=20, first_settlement_epoch_id=EPOCH,
        last_legacy_epoch_id=EPOCH - 1,
    )
    monkeypatch.setattr(wiring.chain_module, "finalized_epoch_snapshot", lambda _: snapshot)
    return adapter


@pytest.mark.parametrize("clock", [False, True], ids=["legacy-previous", "weekly-previous"])
def test_actual_chain_adapter_prefixed_genesis_accepts_the_same_signed_chain(monkeypatch, clock):
    service, data, signer = _harness()
    previous = _previous(data, signer, clock=clock)
    before = deepcopy(previous)
    service._config.chain = _actual_adapter(monkeypatch)
    assert service._config.chain.accepted_weight_epoch_scope()["genesis_hash"] == "0x" + previous["genesis_hash"]
    state = service.public_weight_state(EPOCH)["state"]
    assert state["genesis_hash"] == previous["genesis_hash"] == "1" * 64
    assert previous == before
    assert state["weekly_slot_decay"]["slots"] == [{
        "start_epoch": SOURCE_EPOCH,
        "source_normalized_sha256": data.submission["source_normalized_sha256"],
    }] * 3
    arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    )
    assert len(data.publish_calls) == 1


@pytest.mark.parametrize("clock", [False, True])
@pytest.mark.parametrize("changed", [
    {"genesis": "2" * 64}, {"network": "testnet"}, {"netuid": 72}, {"epoch": EPOCH + 1},
])
def test_actual_adapter_still_rejects_a_different_previous_state_scope(monkeypatch, clock, changed):
    service, data, signer = _harness()
    _previous(data, signer, clock=clock, **changed)
    service._config.chain = _actual_adapter(monkeypatch)
    with pytest.raises(ServiceError, match="reward_clock_previous_invalid"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == data.object_calls == []


@pytest.mark.parametrize("genesis", [None, "", "0x1234", "0x" + "g" * 64, "0x" + "1" * 65])
def test_malformed_chain_genesis_cannot_pass_scope_normalization(monkeypatch, genesis):
    service, data, signer = _harness()
    _previous(data, signer)
    adapter = _actual_adapter(monkeypatch)
    service._config.chain.accepted_weight_epoch_scope = lambda: {
        **adapter.accepted_weight_epoch_scope(), "genesis_hash": genesis,
    }
    with pytest.raises(ServiceError, match="accepted_weight_epoch_scope_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == data.object_calls == []


def test_actual_adapter_still_rejects_a_noncurrent_requested_epoch(monkeypatch):
    service, data, signer = _harness()
    _previous(data, signer)
    service._config.chain = _actual_adapter(monkeypatch)
    with pytest.raises(ServiceError, match="accepted_weight_epoch_not_current"):
        service.public_weight_state(EPOCH + 1)
    assert data.publish_calls == data.object_calls == []


def test_issue_epoch_clock_from_signed_history_and_verified_frozen_archive():
    service, data, signer = _harness()
    state = service.public_weight_state(EPOCH)["state"]
    assert state == data.saved
    assert [clock["start_epoch"] for clock in state["weekly_slot_decay"]["slots"]] == [SOURCE_EPOCH] * 3
    assert len(data.object_calls) == len(data.submission_calls) == 1
    assert data.history_calls[0]["limit"] == 201
    assert data.history_calls[0]["public_only"] is True
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == 15_625_000
    assert vector["burned_residual_ppb"] == 984_375_000
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]


def test_default_disabled_preserves_normal_weights_and_does_no_clock_work():
    service, data, _ = _harness(enabled=False)
    state = service.public_weight_state(EPOCH)["state"]
    assert "weekly_slot_decay" not in state
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 31_250_000
    assert data.object_calls == data.history_calls == []


def test_existing_state_is_returned_unchanged_before_new_clock_or_archive_work():
    service, data, signer = _harness()
    state = _previous(data, signer, clock=False, epoch=EPOCH)
    data.existing = data.previous
    assert service.public_weight_state(EPOCH)["state"] == state
    assert data.object_calls == data.history_calls == data.publish_calls == []


def test_same_epoch_latest_race_returns_existing_signed_bytes_unchanged():
    service, data, signer = _harness()
    state = _previous(data, signer, clock=False, epoch=EPOCH)
    assert service.public_weight_state(EPOCH)["state"] == state
    assert data.object_calls == data.history_calls == data.publish_calls == []


def test_verified_exact_source_checkpoint_needs_no_archives_or_old_history():
    service, data, signer = _harness()
    previous = _previous(data, signer)
    data.payload = b"old archive is unavailable"
    data.history = [{"invalid": "old history is unavailable"}]
    state = service.public_weight_state(EPOCH)["state"]
    assert state["weekly_slot_decay"]["slots"] == previous["weekly_slot_decay"]["slots"]
    assert data.object_calls == data.submission_calls == data.history_calls == []


@pytest.mark.parametrize("field,value", [
    ("round_id", "other-round"), ("submission_id", "other-source"),
    ("miner_hotkey", BURN), ("status", "accepted"), ("is_king", True),
    ("source_size_bytes", 1), ("source_size_bytes", True),
    ("source_archive_sha256", "sha256:" + "f" * 64),
    ("source_normalized_sha256", "sha256:" + "f" * 64),
    ("source_archive_sha256", None), ("source_normalized_sha256", None),
])
def test_source_identity_size_and_both_persisted_digests_must_match(field, value):
    service, data, _ = _harness()
    data.submission[field] = value
    with pytest.raises(ServiceError, match="reward_clock_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == []


def test_mutated_archive_never_acquires_a_signed_reward_clock():
    service, data, _ = _harness()
    data.payload = data.payload[:-1] + bytes([data.payload[-1] ^ 1])
    with pytest.raises(ServiceError, match="reward_clock_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == []


@pytest.mark.parametrize("fault", ["signature", "network", "genesis", "row_epoch", "row_hash"])
def test_previous_accepted_state_requires_valid_signature_and_exact_chain_scope(fault):
    service, data, signer = _harness()
    state = _previous(data, signer, genesis="2" * 64 if fault == "genesis" else "1" * 64)
    if fault == "signature":
        state["signature"]["signature_b64"] = "AAAA"
    elif fault == "network":
        state["network"] = "other"
    elif fault == "row_epoch":
        data.previous["epoch"] -= 1
    elif fault == "row_hash":
        data.previous["state_hash"] = "sha256:" + "f" * 64
    with pytest.raises(ServiceError, match="reward_clock_previous_invalid"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == data.object_calls == []


@pytest.mark.parametrize("fault", ["signature", "scope", "round", "effective", "activation"])
def test_every_history_basis_requires_signature_scope_and_activation_identity(fault):
    service, data, _ = _harness()
    row = data.history[-1]
    if fault == "signature":
        row["reward_basis_doc"]["signature"]["signature_b64"] = "AAAA"
    elif fault == "scope":
        row["arena_netuid"] = 72
    elif fault == "round":
        row["round_id"] = "other"
    elif fault == "effective":
        row["effective_reward_epoch"] += 1
    elif fault == "activation":
        row["reward_activated_at"] = None
    with pytest.raises(ServiceError, match="reward_clock_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == []


def test_current_basis_signature_is_checked_even_when_no_policy_projection_is_needed():
    service, data, _ = _harness()
    data.current["signature"]["signature_b64"] = "AAAA"
    with pytest.raises(ServiceError, match="reward_clock_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == []


def test_archived_signed_history_and_compact_configuration_preserve_tenure():
    service, data, _ = _harness()
    data.history[-1]["round_id"] += "-rewardhistory620"
    for row in data.history:
        row.pop("configuration_doc")
        row.update(cfg_mode='"live"', cfg_baseline_hotkey='"' + BURN + '"')
    state = service.public_weight_state(EPOCH)["state"]
    assert [item["start_epoch"] for item in state["weekly_slot_decay"]["slots"]] == [SOURCE_EPOCH] * 3


@pytest.mark.parametrize("suffix", ["-rewardhistory", "-rewardhistory620-other", "-rewardhistorybad"])
def test_unrecognized_archived_row_alias_cannot_establish_tenure(suffix):
    service, data, _ = _harness()
    data.history[-1]["round_id"] += suffix
    with pytest.raises(ServiceError, match="reward_clock_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == []
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("owner", [ALICE, BOB])
@pytest.mark.parametrize("incorrect_checkpoint", [False, True])
def test_new_qualifying_award_resets_even_with_same_source_or_old_wrong_checkpoint(
    changed, owner, incorrect_checkpoint,
):
    service, data, signer = _harness()
    previous = _previous(data, signer, epoch=29949)
    old_digest = data.submission["source_normalized_sha256"]
    source = ("def run(icp):\n    return icp.name\n" if changed else
              "# Repackaged source with identical safe Python structure.\ndef run(icp):\n    return icp.industry\n")
    payload = archive(source)
    inspected = submission_similarity.inspect_archive(payload)
    fresh = _win(owner, delta=20, start=None, name="new-source", round_id="arena-2026-11-08")
    data.current = signing.sign_document(signer, _basis(
        [deepcopy(fresh) for _ in range(3)], epoch=29950,
        policy=rewards.reward_slot_policy_document(date(2026, 11, 1)),
        round_id=fresh["round_id"],
    ), hash_field="reward_basis_hash")
    row = deepcopy(data.history[0])
    row.update(round_id=fresh["round_id"], effective_reward_epoch=29950,
               reward_basis_doc=data.current)
    data.history.insert(0, row)
    data.submission.update(
        round_id=fresh["round_id"], submission_id=fresh["submission_id"],
        miner_hotkey=owner, source_size_bytes=len(payload), source_archive_sha256=inspected.archive_sha256,
        source_normalized_sha256=inspected.normalized_sha256,
    )
    data.payload = payload
    if incorrect_checkpoint:
        previous = _previous(data, signer, epoch=EPOCH - 1)
    before = deepcopy(previous)
    state = service.public_weight_state(EPOCH)["state"]
    clocks = state["weekly_slot_decay"]["slots"]
    assert len(data.object_calls) == len(data.submission_calls) == 1
    assert len(data.history_calls) == 1
    assert (inspected.normalized_sha256 != old_digest) == changed
    assert [item["start_epoch"] for item in clocks] == [29950] * 3
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 500_000_000
    assert previous["weekly_slot_decay"]["slots"][0]["start_epoch"] == SOURCE_EPOCH
    assert previous == before
    arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    )


def test_truncated_history_preserves_the_signed_award_epoch():
    service, data, _ = _harness()
    data.history = [deepcopy(data.history[0]) for _ in range(201)]
    state = service.public_weight_state(EPOCH)["state"]
    assert [item["start_epoch"] for item in state["weekly_slot_decay"]["slots"]] == [SOURCE_EPOCH] * 3


@pytest.mark.parametrize("mode", ["weekly", "legacy", "missing", "unknown_epoch"])
def test_public_current_reports_only_an_existing_accepted_decay_clock(mode):
    service, data, signer = _harness()
    service._clock = lambda: datetime(2026, 11, 9, tzinfo=timezone.utc)
    service.active_rounds = lambda **kwargs: []
    service.latest_published_round = lambda **kwargs: None
    service._config.chain.current_settlement_epoch = lambda: EPOCH if mode != "unknown_epoch" else None
    if mode in {"weekly", "legacy"}:
        state = _previous(data, signer, clock=mode == "weekly", epoch=EPOCH)
        data.existing = data.previous
    else:
        state = None
    service.public_weight_state = lambda *args: pytest.fail("public view must not issue a state")
    before = deepcopy(data.current)
    public = service.public_current()
    if mode == "weekly":
        assert public["weekly_slot_decay"] == state["weekly_slot_decay"]
        assert public["reward_slot_policy"] == state["reward_basis"]["slot_policy"]
        assert public["reward_slots"] == state["reward_basis"]["reward_slots"]
    else:
        assert "weekly_slot_decay" not in public
    assert data.current == before
    assert data.publish_calls == data.object_calls == data.submission_calls == []


@pytest.mark.parametrize("fault", ["archive", "signature"])
def test_wrong_checkpoint_repair_still_requires_frozen_source_and_signed_history(fault):
    service, data, signer = _harness()
    # A previous issuer attached an older model clock to this exact award.
    bad = {"epochs_per_halving": WEEK, "slots": [
        {"start_epoch": SOURCE_EPOCH - WEEK,
         "source_normalized_sha256": data.submission["source_normalized_sha256"]}
        for _ in range(3)]}
    previous = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71, epoch=EPOCH - 1,
        valid_from_block=BLOCK - 10, valid_until_block=BLOCK, reward_basis=data.current,
        burn_hotkey=BURN, issued_at=data.current["published_at"], weekly_slot_decay=bad,
    )
    data.previous = {key: previous[key] for key in ("network", "netuid", "epoch", "state_hash")}
    data.previous["state_doc"] = previous
    if fault == "archive":
        data.payload = b"invalid frozen archive"
    else:
        data.history[-1]["reward_basis_doc"]["signature"]["signature_b64"] = "AAAA"
    with pytest.raises(ServiceError, match="reward_clock_unavailable"):
        service.public_weight_state(EPOCH)
    assert data.publish_calls == []


@pytest.mark.parametrize("delta,replaced", [(5, [False, False, True]),
                                         (10, [False, True, True]),
                                         (15, [True, True, True])])
@pytest.mark.parametrize("owner", [ALICE, BOB])
def test_total_emissions_service_resets_only_qualifying_tiers_for_same_source(
    monkeypatch, delta, replaced, owner,
):
    service, data, signer = _harness()
    previous = _previous(data, signer, epoch=EPOCH - 51)
    before = deepcopy(previous)
    fresh = _win(owner, delta=delta, start=None, name="same-source-rewin",
                 round_id="arena-2026-11-12")
    policy = deepcopy(rewards.REWARD_SLOT_POLICY_NOV11)
    monkeypatch.setattr(rewards, "reward_slot_policy_document", lambda *_, **__: deepcopy(policy))
    current = _basis([
        deepcopy(fresh if won else slot)
        for won, slot in zip(replaced, data.current["reward_slots"])
    ], epoch=EPOCH - 50, policy=policy, king=owner, round_id=fresh["round_id"])
    data.current = signing.sign_document(signer, current, hash_field="reward_basis_hash")
    row = deepcopy(data.history[0])
    row.update(round_id=fresh["round_id"], effective_reward_epoch=EPOCH - 50,
               reward_basis_doc=data.current)
    data.history.insert(0, row)
    # All new award IDs refer to the exact same frozen source bytes as before.
    data.submission.update(round_id=fresh["round_id"], submission_id=fresh["submission_id"],
                           miner_hotkey=owner)
    state = service.public_weight_state(EPOCH)["state"]
    clocks = state["weekly_slot_decay"]["slots"]
    assert [clock["start_epoch"] for clock in clocks] == [
        EPOCH - 50 if won else SOURCE_EPOCH for won in replaced]
    assert state["reward_basis"]["slot_policy"] == policy
    expected = {}
    for won, share in zip(replaced, [35, 25, 15]):
        hotkey = owner if won else ALICE
        share = Fraction(share, 100 * (1 if won else 32))
        expected[hotkey] = expected.get(hotkey, Fraction(0)) + share
    assert kernel.slot_allocations(state["reward_basis"], EPOCH, HOTKEYS,
                                   until_replaced=True, weekly_slot_decay=state["weekly_slot_decay"]) == expected
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == int(
        sum(expected.values(), Fraction(0)) * 1_000_000_000)
    assert len(data.object_calls) == len(data.submission_calls) == 1
    assert previous == before



def test_already_accepted_epoch_keeps_its_old_wrong_clock_bytes_unchanged():
    service, data, signer = _harness()
    clock = {"epochs_per_halving": WEEK, "slots": [
        {"start_epoch": SOURCE_EPOCH - WEEK,
         "source_normalized_sha256": data.submission["source_normalized_sha256"]}
        for _ in range(3)]}
    accepted = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71, epoch=EPOCH,
        valid_from_block=BLOCK, valid_until_block=BLOCK + 100, reward_basis=data.current,
        burn_hotkey=BURN, issued_at=data.current["published_at"], weekly_slot_decay=clock,
    )
    data.existing = {"state_doc": accepted}
    before = deepcopy(accepted)
    assert service.public_weight_state(EPOCH)["state"] == before
    assert data.object_calls == data.submission_calls == data.history_calls == data.publish_calls == []
