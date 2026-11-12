"""Interrupted replies use existing recovery; no paid call or live DB is used."""
import json
import uuid
from pathlib import Path

import pytest
from tests.lab_arena import owner_natural_provider_expiry686_postgres_test as previous

budget = previous.budget
finite, prior, base, old = budget.finite, budget.prior, budget.base, budget.old
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/689-lab-arena-interrupted-judge-response.sql'
database = previous.database
prepared = previous.prepared
prepared_company = previous.prepared_company
case = previous.case
base_extended = previous.base_extended
extended = previous.extended
provider_case = previous.provider_case
plain_finite = previous.plain_finite
before = previous.before
post = previous.post
installed = previous.installed


@pytest.fixture
def current(installed):
    p = installed
    budget.upgrade(p)
    previous.upgrade(p)
    return p


def transport(p, *, changes=None, omit=None, mismatch=None):
    """The broker's persisted lost-body shape, not an invented HTTP response."""
    cid = 'sha256:' + uuid.uuid4().hex * 2
    call = dict(reason='missing_provider_cost', call_succeeded=False,
                transport_failure=True, transport_error_class='ReadTimeout',
                provider_status=0, observed_provider_status=200,
                openrouter_generation_id='gen-1794458785-fixture',
                credential_fingerprint='sha256:' + 'c' * 64,
                upstream_response_receipt={'broker_observations': {
                    'transport_timing': {'scope': 'initial_provider_transport_send',
                                         'elapsed_ms': 122000}}})
    call.update(changes or {})
    for key in omit or ():
        call.pop(key, None)
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        for kind in ('reservation', 'dispatch', 'uncertain'):
            if kind in (omit or ()):
                continue
            row = dict(entry_kind=kind, run_id=p.parent['run_id'], round_id=old.ROUND,
                       submission_id=p.parent['submission_id'], miner_hotkey=p.parent['miner_hotkey'],
                       stage=p.parent['stage'], call_identity=cid, provider='openrouter',
                       operation_id='openrouter.chat', funding_source=getattr(p, 'funding_source',
                       'host' if p.organizer else 'miner_key'), amount_microusd=100000,
                       entry_doc={'request_hash': 'sha256:' + '4' * 64} if kind == 'reservation'
                                 else {'reason': 'worker_reported', 'call': call} if kind == 'uncertain' else {})
            if kind != 'uncertain':
                row.update(mismatch or {})
            old.insert(q, 'lab_arena_ledger', row)
        q.execute('SET LOCAL session_replication_role=origin')
        entry = old.scalar(q, 'SELECT max(entry_id) FROM lab_arena_ledger WHERE call_identity=%s', (cid,))
    p.c.commit()
    return entry


def proved(p, entry):
    with p.c.cursor() as q:
        value = old.scalar(q, 'SELECT public.lab_arena__retained_provider_transient_v1(l) FROM lab_arena_ledger l WHERE entry_id=%s', (entry,))
    p.c.commit()
    return value


def upgrade(p):
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()


def test_real_lost_body_is_not_recognized_before_fix(current):
    p = current
    budget.diagnostic(p)
    entry = transport(p)
    assert proved(p, entry) is False
    before_rows = base.snapshot(p)
    with pytest.raises(p.pg.Error, match='failed parent'):
        finite.grant(p)
    p.c.rollback()
    assert base.snapshot(p) == before_rows


@pytest.mark.parametrize('error', ['ReadTimeout', 'ReadError', 'RemoteProtocolError'])
def test_complete_broker_binding_proves_lost_response_without_billing_or_data_mutation(current, error):
    p = current
    budget.diagnostic(p)
    entry = transport(p, changes={'transport_error_class': error})
    assert proved(p, entry) is False
    before_rows = base.snapshot(p)
    upgrade(p)
    upgrade(p)
    assert base.snapshot(p) == before_rows
    assert proved(p, entry) is True
    result = finite.grant(p)
    assert result['checkpoints'] == p.checkpoints
    assert result['signal_checkpoints'] == p.signal_checkpoints
    assert base.snapshot(p)['lab_arena_ledger'] == before_rows['lab_arena_ledger']


@pytest.mark.parametrize('fault', ['success', 'missing_transport', 'string_transport', 'no_generation',
                                  'no_fingerprint', 'unknown_error', 'no_reservation', 'no_dispatch',
                                  'cross_round', 'cross_run', 'cross_miner', 'cross_submission',
                                  'cross_stage', 'auth401', 'funding402', 'auth403', 'ambiguous_status'])
def test_unproved_or_credential_failure_does_not_qualify(current, fault):
    p = current
    kwargs = {}
    if fault == 'success': kwargs['changes'] = {'call_succeeded': True}
    elif fault == 'missing_transport': kwargs['omit'] = ['transport_failure']
    elif fault == 'string_transport': kwargs['changes'] = {'transport_failure': 'true'}
    elif fault == 'no_generation': kwargs['omit'] = ['openrouter_generation_id']
    elif fault == 'no_fingerprint': kwargs['omit'] = ['credential_fingerprint']
    elif fault == 'unknown_error': kwargs['changes'] = {'transport_error_class': 'JudgeError'}
    elif fault in ('no_reservation', 'no_dispatch'): kwargs['omit'] = [fault[3:]]
    elif fault.startswith('cross_'):
        kwargs['mismatch'] = {dict(cross_round='round_id', cross_run='run_id', cross_miner='miner_hotkey',
                                  cross_submission='submission_id', cross_stage='stage')[fault]: 1 if fault == 'cross_stage' else 'different'}
    else: kwargs['changes'] = {'observed_provider_status': dict(auth401=401, funding402=402, auth403=403, ambiguous_status=0)[fault]}
    entry = transport(p, **kwargs)
    upgrade(p)
    assert proved(p, entry) is False


def make_failed_budget_child(p):
    child = finite.grant(p)['child_identity']['run_id']
    prior.claim_child(p, child)
    prior.complete_failed(p, child)
    prior.saved_parent(p, child)
    budget.diagnostic(p)
    return transport(p)


def test_owner_can_replace_one_interrupted_exception_and_preserve_all_work(current):
    p = current
    entry = make_failed_budget_child(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, request_id='replacement')
    p.c.rollback()
    upgrade(p)
    assert proved(p, entry) is True
    preview = finite.grant(p, request_id='replacement', approval='approval:replacement', dry=True)
    assert base.snapshot(p) == rows
    result = finite.grant(p, request_id='replacement', approval='approval:replacement')
    assert result['child_identity'] == preview['child_identity']
    after = base.snapshot(p)
    assert len(after['lab_arena_runs']) == len(rows['lab_arena_runs']) + 1
    assert len(after['lab_arena_scoring_supersessions']) == len(rows['lab_arena_scoring_supersessions']) + 1
    originals = {r['run_id']: r for r in rows['lab_arena_runs']}
    assert {r['run_id']: r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    child = result['child_identity']['run_id']
    lease = prior.claim_child(p, child)
    assert lease['company_judgment_cache']['misses'] == []
    prior.complete_failed(p, child)
    old.rpc(p, 'lab_arena_expire_leases(%s)', (old.ROUND,))
    prior.saved_parent(p, child)
    budget.diagnostic(p)
    transport(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, request_id='no-third-exception', approval='approval:no-third')
    p.c.rollback()
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('fault', ['citation_error', 'no_unresolved_failure', 'healed', 'same_approval'])
def test_replacement_rejects_nontransport_or_reused_authority(current, fault):
    p = current
    make_failed_budget_child(p)
    if fault == 'citation_error':
        budget.diagnostic(p, stage='scorer', error='judge_error', reason='invalid_citation')
    elif fault == 'no_unresolved_failure':
        budget.diagnostic(p, unresolved=0)
    elif fault == 'healed':
        budget.receipt(p, status=200, success=True)
    upgrade(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        if fault == 'same_approval':
            finite.grant(p, request_id='replacement')
        else:
            finite.grant(p, request_id='replacement', approval='approval:replacement')
    p.c.rollback()
    assert base.snapshot(p) == rows

