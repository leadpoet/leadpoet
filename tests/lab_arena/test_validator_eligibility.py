"""Stake boundaries and the actual service gate, including capacity planning."""

from types import SimpleNamespace

import pytest

from lab_arena import chain, driver
from lab_arena.service import ServiceError
from gateway.utils.hotkey_roles import (
    MIN_VALIDATOR_STAKE_WEIGHT, ValidatorIneligible, classify_hotkey_from_metagraph, validator_uid,
)
from tests.lab_arena.test_lab_arena_service_rules import _runner_claim_service, _runner_snapshot


HOTKEY = "5" * 48


@pytest.mark.parametrize("offset,eligible", [(-1, False), (0, True), (1, True)])
def test_decoded_chain_stake_boundary_without_rounding(offset, eligible):
    stake = chain._runtime_stake_tao(MIN_VALIDATOR_STAKE_WEIGHT * 10**9 + offset)
    snapshot = _runner_snapshot(HOTKEY, stake=stake)
    if eligible:
        assert validator_uid(snapshot, HOTKEY, netuid=71, network_name="finney") == 0
    else:
        with pytest.raises(ValidatorIneligible, match="runner_stake_below_minimum"):
            validator_uid(snapshot, HOTKEY, netuid=71, network_name="finney")


def test_mainnet_service_threshold_has_no_activity_bypass():
    service = _runner_claim_service(registered=True, role="validator")
    service._config.network_name, service._config.netuid = "finney", 71
    snapshot = _runner_snapshot(HOTKEY, stake=74_999)
    service._config.chain.metagraph = lambda *, finalized: snapshot
    with pytest.raises(ServiceError, match="runner_stake_below_minimum") as error:
        service.handle_claim({})
    assert error.value.status == 403
    snapshot.stake = (75_000,)
    assert snapshot.active == (False,)
    assert service.handle_claim({}) == {"status": "empty"}
    snapshot.stake = (74_999,)
    with pytest.raises(ServiceError, match="runner_stake_below_minimum"):
        service.handle_claim({})


@pytest.mark.parametrize("active,permit", [(True, False), (False, True)])
def test_testnet_claims_keep_shared_gateway_policy(active, permit):
    service = _runner_claim_service(registered=True, role="validator")
    service._config.network_name, service._config.netuid = "test", 401
    snapshot = _runner_snapshot(HOTKEY, stake=0.018, permit=permit)
    snapshot.netuid, snapshot.active = 401, (active,)
    service._config.chain.metagraph = lambda *, finalized: snapshot
    assert service.handle_claim({}) == {"status": "empty"}


@pytest.mark.parametrize("network,netuid", [("finney", 71), ("test", 401)])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("permit", [False, True])
@pytest.mark.parametrize("stake", [0.018, 74_999.999999999, 75_000, 75_000.000000001, 500_001])
def test_claim_capacity_and_gateway_share_one_validator_rule(network, netuid, active, permit, stake):
    service = _runner_claim_service(registered=True, role="validator")
    service._config.network_name, service._config.netuid = network, netuid
    snapshot = _runner_snapshot(HOTKEY, stake=stake, permit=permit)
    snapshot.netuid, snapshot.active = netuid, (active,)
    service._config.chain.metagraph = lambda *, finalized: snapshot
    service._config.defaults = SimpleNamespace(runner_hotkeys=(HOTKEY,))
    service._config.banned_hotkeys_source = lambda: []
    registered, role = classify_hotkey_from_metagraph(HOTKEY, snapshot, network_name=network)
    assert registered
    if role == "validator":
        assert service.handle_claim({}) == {"status": "empty"}
        assert service.runner_settings() == ([HOTKEY], [])
    else:
        with pytest.raises(ServiceError) as error:
            service.handle_claim({})
        assert error.value.status == 403
        with pytest.raises(ServiceError, match="daily_runner_capacity_insufficient"):
            service.runner_settings()


@pytest.mark.parametrize("patch", [
    {"stake": None}, {"stake": ()}, {"stake": (75_000, 75_000)},
    *({"stake": (value,)} for value in [True, "75000", -1, float("nan"), float("inf"), -float("inf")]),
    {"validator_permit": None}, {"validator_permit": ("true",)},
    {"validator_permit": ()}, {"coldkeys": ()}, {"coldkeys": (None,)},
    {"hotkeys": (HOTKEY, HOTKEY)}, {"netuid": 72}, {"netuid": True},
])
def test_bad_snapshot_cannot_allocate_even_with_an_authorizer_override(patch):
    service = _runner_claim_service(registered=True, role="validator")
    snapshot = _runner_snapshot(HOTKEY)
    snapshot.__dict__.update(patch)
    service._config.chain.metagraph = lambda *, finalized: snapshot
    service._store.claim_assignment = lambda **kwargs: pytest.fail("invalid claim allocated")
    service._lease_token = lambda validated: pytest.fail("invalid claim issued token")
    with pytest.raises(ServiceError, match="runner_benchmark_eligibility_unavailable") as error:
        service.handle_claim({})
    assert error.value.status == 503


def test_claim_uses_one_snapshot_for_permission_stake_and_coldkey_exclusions():
    service = _runner_claim_service(registered=True, role="validator")
    snapshot = _runner_snapshot(HOTKEY)
    snapshot.hotkeys = (HOTKEY, "same-owner-miner", "other-miner")
    snapshot.coldkeys = ("owner", "owner", "other")
    snapshot.stake = (75_000, 0, 0)
    snapshot.validator_permit = (True, False, False)
    snapshot.active = (False, True, True)
    reads, claims = [], []
    service._config.chain.metagraph = lambda *, finalized: reads.append(finalized) or snapshot
    service._config.chain.hotkeys_owned_by_same_coldkey = lambda _: pytest.fail("second snapshot")
    service._store.claim_assignment = lambda **kwargs: claims.append(kwargs) or {"status": "empty"}
    assert service.handle_claim({}) == {"status": "empty"}
    assert reads == [True]
    assert claims[0]["excluded_miner_hotkeys"] == [HOTKEY, "same-owner-miner"]


def test_chain_unavailability_refuses_claim_and_does_not_call_store():
    service = _runner_claim_service(registered=True, role="validator")
    def unavailable(**kwargs):
        raise chain.ArenaChainError("RPC unavailable")
    service._config.chain.metagraph = unavailable
    service._store.claim_assignment = lambda **kwargs: pytest.fail("unavailable claim allocated")
    with pytest.raises(ServiceError, match="runner_benchmark_eligibility_unavailable"):
        service.handle_claim({})


def test_mainnet_permit_required_even_for_high_stake_and_activity():
    snapshot = _runner_snapshot(HOTKEY, permit=False, stake=1_000_000)
    snapshot.active = (True,)
    with pytest.raises(ValidatorIneligible, match="runner_validator_required"):
        validator_uid(snapshot, HOTKEY, netuid=71, network_name="finney")


def test_existing_work_identity_does_not_require_stake_or_activity():
    service = _runner_claim_service(registered=True, role="validator")
    snapshot = _runner_snapshot(HOTKEY, stake=1)
    snapshot.active, snapshot.stake = None, None
    service._config.validator_authorizer = None
    service._config.chain.metagraph = lambda *, finalized: snapshot
    service._require_validator_authority(HOTKEY)
    snapshot.validator_permit = (False,)
    with pytest.raises(ServiceError, match="runner_validator_required"):
        service._require_validator_authority(HOTKEY)


def _capacity_service():
    service = _runner_claim_service(registered=True, role="validator")
    snapshot = _runner_snapshot(HOTKEY)
    snapshot.hotkeys = (HOTKEY, "low", "miner", "unplanned")
    snapshot.coldkeys = ("owner", "low-owner", "miner-owner", "unplanned-owner")
    snapshot.stake = (75_000, 74_999, 1_000_000, 100_000)
    snapshot.validator_permit = (True, True, False, True)
    snapshot.active = (False, True, True, False)
    service._config.chain.metagraph = lambda *, finalized: snapshot
    service._config.defaults = SimpleNamespace(runner_hotkeys=(HOTKEY, HOTKEY, "low", "miner", "absent"))
    service._config.banned_hotkeys_source = lambda: []
    return service, snapshot


def test_capacity_counts_only_eligible_planned_runners():
    service, snapshot = _capacity_service()
    assert service.runner_settings() == ([HOTKEY], [])
    assert service._benchmark_validator_uid(snapshot, "unplanned") == 3
    service._config.defaults.runner_hotkeys = ("low", "miner", "absent")
    with pytest.raises(ServiceError, match="daily_runner_capacity_insufficient"):
        service.runner_settings()


def test_capacity_failure_does_not_stop_active_rounds_or_rewards():
    service, snapshot = _capacity_service()
    snapshot.stake = (0, 0, 0, 0)
    effects = []
    service.promote_pending_baselines = lambda: {"promoted": 0}
    service.active_rounds = lambda: [{"round_id": "old", "status": "stage1"}]
    service.advance_round = lambda round_id: effects.append(round_id)
    service.ensure_daily_round = service.runner_settings
    service.activate_pending_rewards = lambda: effects.append("rewards") or {"activated": 1}
    assert "failed ensure_daily_round" in driver.drive_once(service)
    assert effects == ["old", "rewards"]


def test_capacity_malformed_stake_is_unavailable_not_silently_excluded():
    service, snapshot = _capacity_service()
    snapshot.stake = (75_000, float("nan"), 0, 100_000)
    with pytest.raises(ServiceError, match="runner_benchmark_eligibility_unavailable"):
        service.runner_settings()
