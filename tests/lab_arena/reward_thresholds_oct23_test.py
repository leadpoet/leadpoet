"""New signed epoch projections change thresholds without rewriting round history."""
from copy import deepcopy
from datetime import date, datetime, timezone
from fractions import Fraction
from types import SimpleNamespace

import pytest

from lab_arena import rewards, signing, weight_state
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStoreError
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.slot_decay_kernel_test import ALICE, BOB, BURN, HOTKEYS, _basis, _win
from tests.lab_arena.reward_shares_oct22_test import _activation_service
from tests.lab_arena.slot_decay_postgres_test import database, state, _winner
from tests.lab_arena.reward_slots_postgres_test import _basis as _pg_basis

CUTOVER = 25954


def _policy():
    return rewards.reward_slot_policy_document(date(2026, 10, 23))


@pytest.mark.parametrize('delta,occupied,share', [
    (7.999, [False, False, False], 0),
    (8, [False, False, True], Fraction(3, 20)),
    (14.999, [False, False, True], Fraction(3, 20)),
    (15, [False, True, True], Fraction(3, 10)),
    (19.999, [False, True, True], Fraction(3, 10)),
    (20, [True, True, True], Fraction(1, 2)),
])
def test_exact_new_threshold_boundaries(delta, occupied, share):
    policy = _policy()
    assert [t['minimum_improvement'] for t in policy['tiers']] == [20, 15, 8]
    win = _win(ALICE, delta=delta, start=CUTOVER)
    basis = _basis([deepcopy(win) if present else None for present in occupied],
                   epoch=CUTOVER, policy=policy)
    allocation = kernel.slot_allocations(basis, CUTOVER, HOTKEYS, until_replaced=True)
    assert allocation == ({ALICE: share} if share else {})
    for index, present in enumerate(occupied):
        if not present:
            invalid = deepcopy(basis['reward_slots'])
            invalid[index] = deepcopy(win)
            with pytest.raises(kernel.LabArenaRewardError, match='threshold'):
                kernel.validate_reward_slots(invalid, policy, effective_reward_epoch=CUTOVER)


def _service(*, existing=None, epoch=CUTOVER, race=False):
    signer = signing.LocalSigner.generate()
    old_policy = rewards.reward_slot_policy_document(date(2026, 10, 22))
    high = _win(ALICE, delta=11.1, start=25846, name='oct18', round_id='arena-2026-10-18')
    low = _win(BOB, delta=6.0454545, start=None, name='oct22', round_id='arena-2026-10-22')
    original = signing.sign_document(signer, _basis(
        [high, low, deepcopy(low)], epoch=25937, policy=old_policy,
        king=BOB, round_id='arena-2026-10-22'), hash_field='reward_basis_hash')
    saved = {} if existing is None else {epoch: existing}
    calls = []
    def snapshot(rid, policy):
        calls.append((rid, deepcopy(policy)))
        return [None, None, deepcopy(high)]
    def publish(network, netuid, requested, digest, state):
        if race:
            # Another gateway signed the same projected body independently.
            signed_basis = signing.sign_document(signer, state['reward_basis'], hash_field='reward_basis_hash')
            winner = weight_state.build_accepted_weight_state(
                signer, network=network, netuid=netuid, genesis_hash='1'*64,
                epoch=requested, valid_from_block=1000, valid_until_block=1100,
                reward_basis=signed_basis, burn_hotkey=BURN, issued_at=original['published_at'])
            assert winner['state_hash'] != digest
            saved[requested] = {'state_doc': winner}
            raise ArenaStoreError('lab_arena_weight_state_conflict')
        saved[requested] = {'state_doc': deepcopy(state)}
        return {'state': state}
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode='live', accepted_burn_hotkey=BURN,
        chain=SimpleNamespace(current_settlement_epoch=lambda: epoch,
            accepted_weight_epoch_scope=lambda: {'epoch': epoch, 'genesis_hash': '1'*64,
                'valid_from_block': 1000, 'valid_until_block': 1100}))
    service._store = SimpleNamespace(
        published_reward_bases=lambda **kwargs: [{'configuration_doc': {'mode': 'live',
            'baseline_hotkey': BURN}, 'reward_basis_doc': original, 'status': 'published', 'round_id': original['round_id']}],
        get_weight_state=lambda network, netuid, requested: saved.get(requested),
        reward_slot_snapshot=snapshot, publish_weight_state=publish)
    service._reward_signer = lambda: signer
    service._chain_scope = lambda: ('finney', 71)
    service._clock = lambda: datetime(2026, 10, 23, tzinfo=timezone.utc)
    service.active_rounds = lambda **kwargs: []
    service.latest_published_round = lambda **kwargs: None
    return service, original, saved, calls, signer


def test_current_projection_pays_fifteen_and_preserves_signed_source_and_age():
    service, original, saved, calls, signer = _service()
    before = deepcopy(original)
    state = service.public_weight_state(CUTOVER)['state']
    basis = state['reward_basis']
    assert basis['slot_policy'] == _policy()
    assert basis['reward_slots'][:2] == [None, None]
    assert basis['reward_slots'][2]['start_epoch'] == 25846
    assert {k:v for k,v in basis.items() if k not in ('slot_policy','reward_slots','signature','reward_basis_hash')} == {
        k:v for k,v in before.items() if k not in ('slot_policy','reward_slots','signature','reward_basis_hash')}
    assert original == before
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector['champion_share_ppb'] == 150_000_000
    assert vector['burned_residual_ppb'] == 850_000_000
    assert vector['sparse_uids'] == [0, 1]
    assert service.public_weight_state(CUTOVER)['state'] == state
    assert arena_weights.verify_accepted_weight_state_signature(state,
        public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash) == state['state_hash']
    current = service.public_current()
    assert current['reward_slot_policy'] == _policy()
    assert current['reward_slots'] == basis['reward_slots']


def test_pre_cutover_state_and_basis_are_unchanged():
    service, original, saved, calls, signer = _service(epoch=CUTOVER-1)
    state = service.public_weight_state(CUTOVER-1)['state']
    assert state['reward_basis'] == original
    assert calls == []
    service._config.chain.accepted_weight_epoch_scope = lambda: pytest.fail('historical state needs no fresh scope')
    assert service.public_weight_state(CUTOVER-1)['state'] == state


def test_original_signature_must_be_verified_before_re_signing():
    service, original, saved, calls, signer = _service()
    original['signature']['signature_b64'] = 'AAAA'
    with pytest.raises(ServiceError, match='reward_history_signature_invalid'):
        service.public_weight_state(CUTOVER)
    assert saved == {}


def test_projection_race_returns_persisted_valid_state():
    service, original, saved, calls, signer = _service(race=True)
    state = service.public_weight_state(CUTOVER)['state']
    assert state == saved[CUTOVER]['state_doc']
    assert arena_weights.derive_arena_weights(state, HOTKEYS)['champion_share_ppb'] == 150_000_000


def test_pending_old_round_uses_new_thresholds_after_cutover():
    service, row, captured = _activation_service('2026-10-22')
    service._config.chain.current_settlement_epoch = lambda: CUTOVER-1
    assert service.activate_reward(row['round_id'])['status'] == 'activated'
    assert captured['basis']['slot_policy'] == _policy()


def test_source_round_null_clock_keeps_original_epoch_and_halving():
    service, original, saved, calls, signer = _service(epoch=CUTOVER+140)
    # A current source in SQL is NULL even after activation. The projection
    # must retain the original basis epoch, not restart the achievement clock.
    source = _win(BOB, delta=11.1, start=None, round_id=original['round_id'])
    body = {k:v for k,v in original.items() if k not in ('signature','reward_basis_hash')}
    body['reward_slots'] = [deepcopy(source), deepcopy(source), deepcopy(source)]
    original.clear()
    original.update(signing.sign_document(signer, rewards.finalize_reward_basis(body), hash_field='reward_basis_hash'))
    state = service.public_weight_state(CUTOVER+140)['state']
    assert state['reward_basis']['effective_reward_epoch'] == 25937
    assert state['reward_basis']['reward_slots'][2]['start_epoch'] is None
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector['champion_share_ppb'] == 75_000_000


def test_race_cannot_accept_different_burn_or_validity_body():
    service, original, saved, calls, signer = _service()
    def bad_publish(network, netuid, epoch, digest, candidate):
        winner = weight_state.build_accepted_weight_state(
            signer, network=network, netuid=netuid, genesis_hash='1'*64,
            epoch=epoch, valid_from_block=1000, valid_until_block=1101,
            reward_basis=candidate['reward_basis'], burn_hotkey=BURN,
            issued_at=original['published_at'])
        saved[epoch] = {'state_doc': winner}
        raise ArenaStoreError('lab_arena_weight_state_conflict')
    service._store.publish_weight_state = bad_publish
    with pytest.raises(ServiceError, match='accepted_weight_state_conflict'):
        service.public_weight_state(CUTOVER)


def test_race_cannot_accept_invalid_signature():
    service, original, saved, calls, signer = _service()
    def bad_publish(network, netuid, epoch, digest, candidate):
        candidate['signature']['signature_b64'] = 'AAAA'
        saved[epoch] = {'state_doc': candidate}
        raise ArenaStoreError('lab_arena_weight_state_conflict')
    service._store.publish_weight_state = bad_publish
    with pytest.raises(ServiceError, match='accepted_weight_state_invalid'):
        service.public_weight_state(CUTOVER)


def test_postgres_snapshot_requalifies_current_and_future_without_history_write(state):
    store, control, chain = state
    high = _winner(control, chain, 'oct18', day='2026-10-18', delta=11.1,
                   hotkey=ALICE, epoch=25846)
    current = _winner(control, chain, 'oct22', day='2026-10-22', delta=6.0454545,
                      hotkey=BOB, activated=False)
    old_policy = rewards.reward_slot_policy_document(date(2026, 10, 22))
    slots = store.reward_slot_snapshot(current, old_policy)
    assert [s['round_id'] for s in slots] == [high, current, current]
    old_basis, key = _pg_basis(current, slots, policy=old_policy, hotkey=BOB, epoch=25937)
    assert store.activate_reward(current, old_basis, key)['status'] == 'activated'
    original = deepcopy(store.get_round(current)['reward_basis_doc'])
    revised = store.reward_slot_snapshot(current, _policy())
    assert revised[:2] == [None, None]
    assert revised[2]['round_id'] == high
    assert revised[2]['start_epoch'] == 25846
    future = _winner(control, chain, 'future', day='2026-10-23', delta=15,
                     hotkey=ALICE, activated=False)
    future_slots = store.reward_slot_snapshot(future, _policy())
    assert [s['round_id'] if s else None for s in future_slots] == [None, future, future]
    future_basis, key = _pg_basis(future, future_slots, policy=_policy(), hotkey=ALICE, epoch=CUTOVER)
    assert store.activate_reward(future, future_basis, key)['status'] == 'activated'
    assert store.get_round(current)['reward_basis_doc'] == original
    assert kernel.slot_allocations(future_basis, CUTOVER, HOTKEYS, until_replaced=True) == {ALICE: Fraction(3, 10)}


def test_existing_cutover_state_returns_exact_old_policy_without_snapshot_or_signing():
    service, original, saved, calls, signer = _service()
    accepted = weight_state.build_accepted_weight_state(
        signer, network='finney', netuid=71, genesis_hash='1'*64,
        epoch=CUTOVER, valid_from_block=1000, valid_until_block=1100,
        reward_basis=original, burn_hotkey=BURN, issued_at=original['published_at'])
    saved[CUTOVER] = {'state_doc': accepted}
    service._store.reward_slot_snapshot = lambda *args: pytest.fail('accepted state needs no new snapshot')
    service._sign = lambda *args: pytest.fail('accepted state must not be re-signed')
    assert service.public_weight_state(CUTOVER)['state'] == accepted
    assert service.public_current()['reward_slot_policy'] == original['slot_policy']


def test_accepted_projected_state_survives_reopened_source_without_new_snapshot():
    service, original, saved, calls, signer = _service()
    accepted = service.public_weight_state(CUTOVER)['state']
    service._store.reward_slot_snapshot = lambda *args: pytest.fail('reopened source must not rewrite accepted state')
    service._sign = lambda *args: pytest.fail('accepted state must not be re-signed')
    assert service.public_weight_state(CUTOVER)['state'] == accepted
    assert service.public_current()['reward_slots'] == accepted['reward_basis']['reward_slots']


def test_reopened_source_uses_signed_authority_without_snapshot():
    service, original, saved, calls, signer = _service()
    service._store.reward_slot_snapshot = lambda *args: pytest.fail('reopened source must use signed holders')
    accepted = service.public_weight_state(CUTOVER)['state']
    assert arena_weights.derive_arena_weights(accepted, HOTKEYS)['champion_share_ppb'] == 150_000_000


def test_newer_lower_slot_eight_replaces_older_eleven_without_old_owner_resurrection():
    service, original, saved, calls, signer = _service()
    newest = _win(BOB, delta=8, start=None, round_id=original['round_id'])
    body = {k:v for k,v in original.items() if k not in ('signature','reward_basis_hash')}
    body['reward_slots'][1:] = [newest, deepcopy(newest)]
    original.clear()
    original.update(signing.sign_document(signer, rewards.finalize_reward_basis(body), hash_field='reward_basis_hash'))
    service._store.reward_slot_snapshot = lambda *args: pytest.fail('displaced history must not return')
    state = service.public_weight_state(CUTOVER)['state']
    assert state['reward_basis']['reward_slots'] == [None, None, newest]
    assert arena_weights.derive_arena_weights(state, HOTKEYS)['sparse_uids'] == [0, 2]


def test_twenty_point_signed_holder_qualifies_for_all_new_tiers():
    service, original, saved, calls, signer = _service()
    winner = _win(ALICE, delta=20, start=25846)
    body = {k:v for k,v in original.items() if k not in ('signature','reward_basis_hash')}
    body['reward_slots'] = [deepcopy(winner) for _ in range(3)]
    original.clear()
    original.update(signing.sign_document(signer, rewards.finalize_reward_basis(body), hash_field='reward_basis_hash'))
    state = service.public_weight_state(CUTOVER)['state']
    assert state['reward_basis']['reward_slots'] == [winner] * 3
    assert arena_weights.derive_arena_weights(state, HOTKEYS)['champion_share_ppb'] == 500_000_000


def test_wrong_key_source_cannot_gain_new_reward_authority():
    service, original, saved, calls, signer = _service()
    original.update(signing.sign_document(signing.LocalSigner.generate(), original, hash_field='reward_basis_hash'))
    with pytest.raises(ServiceError, match='reward_history_signature_invalid'):
        service.public_weight_state(CUTOVER)
    assert saved == {}


def test_unsupported_assignment_or_decay_cannot_be_projected():
    service, original, saved, calls, signer = _service()
    body = {k:v for k,v in original.items() if k not in ('signature','reward_basis_hash')}
    body['slot_policy']['assignment_mode'] = 'highest_only'
    body['reward_slots'][2] = None
    highest = rewards.finalize_reward_basis(body)
    with pytest.raises(kernel.LabArenaRewardError, match='increasing all_qualifying'):
        rewards.reward_basis_policy_projection_document(highest, _policy())
    body['slot_policy']['assignment_mode'] = 'all_qualifying'
    body['slot_policy']['decay']['epochs_per_halving'] = 141
    different_decay = rewards.finalize_reward_basis(body)
    with pytest.raises(kernel.LabArenaRewardError, match='unchanged decay'):
        rewards.reward_basis_policy_projection_document(different_decay, _policy())


@pytest.mark.parametrize('field,value', [
    ('miner_hotkey', BOB), ('submission_id', 'different-source'), ('start_epoch', 25845),
])
def test_validly_signed_but_unauthorized_projected_holder_is_rejected(field, value):
    service, original, saved, calls, signer = _service()
    projected = rewards.reward_basis_policy_projection_document(original, _policy())
    projected['reward_slots'][2][field] = value
    projected = rewards.finalize_reward_basis({
        k:v for k,v in projected.items() if k != 'reward_basis_hash'})
    signed = signing.sign_document(signer, projected, hash_field='reward_basis_hash')
    accepted = weight_state.build_accepted_weight_state(
        signer, network='finney', netuid=71, genesis_hash='1'*64,
        epoch=CUTOVER, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=original['published_at'])
    saved[CUTOVER] = {'state_doc': accepted}
    with pytest.raises(ServiceError, match='accepted_weight_state_reward_conflict'):
        service.public_weight_state(CUTOVER)
    with pytest.raises(ServiceError, match='accepted_weight_state_reward_conflict'):
        service.public_current()
