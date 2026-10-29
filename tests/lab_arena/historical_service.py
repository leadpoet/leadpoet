"""Explicit service boundaries for exact historical SQL migration exercises."""

from datetime import date


def pin_pre484_credit_terminal(service, connect, *, round_id):
    """Deny terminal-credit authority in an explicitly pinned pre-484 SQL test.

    This does not disable baseline recovery or replace an existing RPC. Current
    schemas must use their real terminal-credit predicate.
    """
    from lab_arena import contracts

    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena__miner_credit_judgment_terminal_v1(text)')")
            assert cursor.fetchone()[0] is None, 'pre484 service cannot replace terminal-credit authority'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    assert row is not None
    configuration_hash = contracts.document_hash(row['configuration_doc'])

    def historical_terminal_credit_boundary(source_id):
        source = service._store.get_run(source_id)
        assert source is not None and source['round_id'] == round_id
        current = service._store.get_round(round_id)
        assert current is not None and contracts.document_hash(current['configuration_doc']) == configuration_hash
        return False

    service._terminal_miner_credit_judgment = historical_terminal_credit_boundary
    return service


def start_pre470_round(harness, *, round_id, epoch):
    """Admit a historical SQL exercise using a live disposable wall-clock window."""
    from datetime import datetime, timedelta, timezone

    assert date.fromisoformat(round_id[6:16]) < date(2026, 10, 14)
    harness.chain.epoch = epoch
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.create_round(harness.clock.now + timedelta(minutes=30), round_id=round_id)
    harness.round_id = round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()['submission_cutoff'])
    assert harness.service.advance_round(round_id)['status'] == 'ok'
    pin_pre470_baseline_service(harness.service, harness.connect, round_id=round_id)
    participants = harness.service.store.get_round(round_id)['participants']
    for participant in participants:
        harness.flavors.setdefault(participant['submission_id'], 'PublicBaseline')
    return len(participants)


def pin_pre470_baseline_service(service, connect, *, round_id):
    """Keep pre-470 migration tests out of the later baseline-recovery branch.

    Migration 470 introduced the recovery RPC and its service hook. A test of
    an earlier frozen migration still uses actual SQL transitions, scoring and
    archive guards; it cannot exercise that later recovery protocol. This
    explicit boundary is forbidden on current schemas and modern rounds.
    """
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_recover_baseline_v1(text,jsonb)')")
            assert cursor.fetchone()[0] is None, 'pre470 service cannot replace current recovery authority'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    assert row is not None
    # Fresh rehearsals use a current wall-clock schedule on the old protocol.
    # The immutable identifier selects that protocol; bind the actual committed
    # evaluation date separately so it cannot drift after this fixture boundary.
    assert date.fromisoformat(round_id[6:16]) < date(2026, 10, 14)
    frozen_date = str(row.get('evaluation_date') or round_id[6:16])[:10]
    date.fromisoformat(frozen_date)
    assert (row.get('configuration_doc') or {}).get('mode') == 'live'

    def historical_baseline_boundary(current):
        assert current['round_id'] == round_id
        assert str(current.get('evaluation_date') or current['round_id'][6:16])[:10] == frozen_date
        # No baseline-recovery RPC existed in this frozen schema. The ordinary
        # historical transition and all of its database guards still execute.

    pin_pre484_credit_terminal(service, connect, round_id=round_id)
    terminal_credit = service._terminal_miner_credit_judgment

    def historical_terminal_credit_boundary(source_id):
        # Terminal miner-credit authority arrived in 484. These old SQL proofs
        # cannot grant it; False preserves the ordinary NULL/incomplete path.
        historical_baseline_boundary(service._store.get_round(round_id))
        return terminal_credit(source_id)

    service._recover_baseline = historical_baseline_boundary
    service._terminal_miner_credit_judgment = historical_terminal_credit_boundary
    return service


def seed_pre428_initial_reward(service, connect, *, round_id, expected_evaluation_date=None):
    """Seed the genuine V1 reward used by a pre-428 migration proof.

    This is fixture setup, not a claim that current slot-based activation runs
    on old SQL. The canonical document, signature and historical activation RPC
    stay real. The construction follows service.py before 1b5d196af (428).
    """
    from lab_arena import rewards

    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_reward_slot_snapshot(text,jsonb)')")
            assert cursor.fetchone()[0] is None, 'pre428 seed cannot replace current reward authority'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    assert date.fromisoformat(round_id[6:16]) < date(2026, 10, 8)
    if expected_evaluation_date is not None:
        assert str(row['evaluation_date']) == str(expected_evaluation_date)
    else:
        assert date.fromisoformat(str(row['evaluation_date'])[:10]) < date(2026, 10, 8)
    assert row['status'] == 'published'
    config = row['configuration_doc']
    assert config['mode'] == 'live' and config['rewards_enabled'] is True
    assert row.get('reward_activated_at') is None
    decision = row['publication_doc']['king_decision']
    assert decision['outcome'] in ('crowned', 'defended', 'retained_ineligible', 'no_king')
    daily_hotkey = decision.get('king_hotkey') if decision['outcome'] == 'crowned' else ''
    if daily_hotkey:
        assert daily_hotkey != config['baseline_hotkey']
        assert not row.get('promotion_required') or row['baseline_promoted_at'] is not None
    assert not service._superseded_by_published_day(row)
    assert not service._pending_promotion_blocks()
    network, netuid = service._chain_scope()
    prior = service._store.published_reward_bases(mode='live', network_name=network, netuid=netuid, limit=200)
    assert not service._usable_reward_bases(prior), 'initial reward seed cannot replace an existing authority'
    maximum = max((int(r['effective_reward_epoch']) for r in prior if r.get('effective_reward_epoch') is not None), default=-1)
    effective = max(int(service.config.chain.current_settlement_epoch()) + 1, maximum + 1)
    basis = service._sign(rewards.reward_basis_document(
        round_id=round_id, published_at=row['publication_doc']['published_at'],
        finalized_epoch=effective - 1, king_outcome='crowned' if daily_hotkey else 'no_king',
        king_hotkey=daily_hotkey or '', reward_constants=config['reward_constants'],
    ), 'reward_basis_hash')
    return service._store.activate_reward(round_id, basis, service.signing_key_document())
