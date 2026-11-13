"""Archived signed authority survives a finite champion reassessment."""
from copy import deepcopy
from datetime import date
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import rewards, signing
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import current_total_emissions_service_database, prepare_current_service_reads
from tests.lab_arena.reward_slots_postgres_test import _insert, _publication, BASELINE
from tests.lab_arena.total_emissions_reward_slots681_postgres_test import POLICY

SQL = (Path(__file__).parents[2] / 'scripts/696-lab-arena-correction-reward-history.sql').read_text()
RPC = 'lab_arena_correction_reward_history_v1'
CHAINS = itertools.count(57500)


@pytest.fixture(scope='module')
def database():
    g = current_total_emissions_service_database()
    pg, dsn = next(g)
    try:
        prepare_current_service_reads(lambda: pg.connect(**dsn))
        with pg.connect(**dsn) as c, c.cursor() as q:
            q.execute(SQL)
        yield pg, dsn
    finally:
        g.close()


@pytest.fixture
def state(database):
    pg, dsn = database
    c = pg.connect(**dsn)
    c.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        yield store, c, next(CHAINS)
    finally:
        store.close()
        c.close()


def basis(rid, epoch=400, *, outcome='no_king', hotkey=''):
    signer = signing.LocalSigner.generate()
    doc = rewards.reward_basis_document(round_id=rid, published_at='2026-10-08T12:00:00Z',
        finalized_epoch=epoch-1, king_outcome=outcome, king_hotkey=hotkey,
        slot_policy=POLICY, reward_slots=[None, None, None])
    return signing.sign_document(signer, doc, hash_field='reward_basis_hash'), signing.signing_key_document(signer.public_key_der)


def seed(state, *, mode='live'):
    store, c, chain = state
    rid = f'arena-2026-11-13-{chain}history'
    pub = _publication(rid, 0)
    pub['king_decision'] = dict(outcome='no_king', king_hotkey='', king_submission_id=None, winner_submission_id=None)
    doc, key = basis(rid)
    with c.cursor() as q:
        q.execute('SET session_replication_role=replica')
    try:
        assert _insert(c, chain, 'history', day='2026-11-13', mode=mode, publication=pub, activated=False) == rid
        with c.cursor() as q:
            q.execute("UPDATE lab_arena_rounds SET reward_basis_doc=%s::jsonb,reward_basis_hash=%s,signing_key_doc=%s::jsonb,effective_reward_epoch=400,reward_activated_at=clock_timestamp(),king_start_epoch=0,configuration_doc=configuration_doc||%s::jsonb WHERE round_id=%s", (json.dumps(doc), doc['reward_basis_hash'], json.dumps(key), json.dumps({'private_probe': 'must remain private', 'schedule': {'final_scoring_close': '2026-11-13T20:30:00Z', 'publication_deadline': '2026-11-13T21:00:00Z'}}), rid))
            # Synthetic completed baseline exercises the real final-authority
            # trigger when the ordinary activation RPC runs below.
            sid = rid + '-baseline'
            q.execute("UPDATE lab_arena_rounds SET participants=%s::jsonb,configuration_doc=configuration_doc||'{\"stage_1_icp_count\":1,\"stage_2_icp_count\":1}' WHERE round_id=%s", (json.dumps([{'submission_id':sid,'miner_hotkey':BASELINE,'is_king':True}]),rid))
            q.execute("INSERT INTO lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king)VALUES(%s,%s,%s,'frozen',true)", (sid,rid,BASELINE))
            for pos in range(2):
                execution = rid + '-execute-' + str(pos)
                for kind in ('execute','score'):
                    q.execute("INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,per_icp_score,output_ref,result_doc,scored_run_id,scoring_generation_floor)VALUES(%s,%s,%s,%s,%s,%s,%s,1,%s,'accepted','accepted',0,'fixture-output','{\"terminal_status\":\"accepted\"}',%s,0)", (execution if kind=='execute' else execution+'-score',execution+'-'+kind,rid,sid,BASELINE,pos+1,pos,kind,None if kind=='execute' else execution))
            q.execute("SELECT to_jsonb(r)-'updated_at' FROM lab_arena_rounds r WHERE round_id=%s", (rid,))
            old = q.fetchone()[0]
    finally:
        with c.cursor() as q:
            q.execute('SET session_replication_role=origin')
    return rid, doc, old


def archive(c, old, *, reopen=True, mutate=None, generation=1):
    new = deepcopy(old)
    new.update(status='stage2_judged', status_generation=old['status_generation']+1, published_at=None, publication_doc=None)
    for key in ('reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at','king_start_epoch','king_outcome','king_hotkey','promotion_doc','baseline_promoted_at'):
        new[key] = None
    proof = dict(operation='round_operational_update_v1', recovery_kind='operator_postpublication_champion_correction_v1', approval_ref='fixture exact seven correction', runtime_image_digest='sha256:'+'c'*64, requests=[{'private_parent': 'never project this'}])
    if mutate:
        mutate(old, new, proof)
    with c.cursor() as q:
        q.execute("SELECT 'sha256:'||encode(extensions.digest(%s::jsonb::text,'sha256'),'hex')", (json.dumps(old),))
        proof['round_sha256'] = q.fetchone()[0]
        q.execute('SET session_replication_role=replica')
        try:
            q.execute("INSERT INTO lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)VALUES(%s,%s,%s,'sha256:'||encode(extensions.digest(%s::jsonb::text,'sha256'),'hex'),'fixture696',txid_current(),pg_backend_pid(),%s::jsonb,%s::jsonb,'{}',%s::jsonb)", (old['round_id'], generation, 'fixture696-'+str(generation), json.dumps(proof), json.dumps(old), json.dumps(new), json.dumps(proof)))
            if reopen:
                q.execute("UPDATE lab_arena_rounds SET status='stage2_judged',status_generation=status_generation+1,published_at=NULL,publication_doc=NULL,reward_basis_doc=NULL,reward_basis_hash=NULL,signing_key_doc=NULL,effective_reward_epoch=NULL,reward_activated_at=NULL,king_start_epoch=NULL,king_outcome=NULL,king_hotkey=NULL,promotion_doc=NULL,baseline_promoted_at=NULL WHERE round_id=%s", (old['round_id'],))
        finally:
            q.execute('SET session_replication_role=origin')
    return new


def read(store, chain, **kw):
    return store.published_reward_bases(mode='live', network_name='finney', netuid=chain, **kw)


def test_old_signed_basis_governs_reopened_round_then_future_basis_only(state):
    store, c, chain = state
    rid, old_basis, old = seed(state)
    before = read(store, chain)
    archive(c, old)
    after = read(store, chain)
    assert after == before
    assert rewards.governing_reward_basis([r['reward_basis_doc'] for r in after], 450) == old_basis
    service = ArenaService.__new__(ArenaService)
    service._store = store
    service._config = SimpleNamespace(mode='live')
    service._chain_scope = lambda: ('finney', chain)
    assert service.public_reward_basis(450) == old_basis
    # Ordinary activation's max/prior inputs use the same archived authority.
    too_early, early_key = basis(rid, 399)
    future, key = basis(rid, 500)
    with c.cursor() as q:
        q.execute('SET session_replication_role=replica')
        q.execute("UPDATE lab_arena_rounds SET status='published',published_at=%s::timestamptz,publication_doc=%s::jsonb,king_outcome='no_king' WHERE round_id=%s", (old['published_at'], json.dumps(old['publication_doc']), rid))
        q.execute('SET session_replication_role=origin')
    assert store.activate_reward(rid, too_early, early_key) == {'status':'epoch_conflict','minimum_effective_reward_epoch':401}
    # This minimal read fixture has no full service publication proof. Keep
    # its final-authority guard, then seed the future activated projection;
    # the seven-job service integration exercises real prospective activation.
    with pytest.raises(ArenaStoreError, match='final_baseline_incomplete'):
        store.activate_reward(rid, future, key)
    with c.cursor() as q:
        q.execute('SET session_replication_role=replica')
        q.execute("UPDATE lab_arena_rounds SET reward_basis_doc=%s::jsonb,reward_basis_hash=%s,signing_key_doc=%s::jsonb,effective_reward_epoch=500,reward_activated_at=clock_timestamp(),king_start_epoch=0 WHERE round_id=%s", (json.dumps(future),future['reward_basis_hash'],json.dumps(key),rid))
        q.execute('SET session_replication_role=origin')
    rows = read(store, chain)
    assert [r['effective_reward_epoch'] for r in rows] == [500, 400]
    assert rewards.governing_reward_basis([r['reward_basis_doc'] for r in rows], 499) == old_basis
    assert rewards.governing_reward_basis([r['reward_basis_doc'] for r in rows], 500) == future
    assert max(r['effective_reward_epoch'] for r in rows)+1 == 501


def test_overlap_dedup_scope_paging_and_public_redaction(state):
    store, c, chain = state
    rid, doc, old = seed(state)
    archive(c, old, reopen=False)
    assert len(read(store, chain)) == 1
    archive(c, old, reopen=False, generation=2)
    assert len(read(store, chain)) == 1
    public = read(store, chain, public_only=True)
    assert public[0]['reward_basis_doc'] == doc
    assert 'private_probe' not in public[0]['configuration_doc']
    assert 'private_parent' not in json.dumps(public)
    assert store.published_reward_bases(mode='shadow', network_name='finney', netuid=chain) == []
    assert store.published_reward_bases(mode='live', network_name='test', netuid=chain) == []
    assert store.published_reward_bases(mode='live', network_name='finney', netuid=chain+100000) == []
    params = dict(p_mode='live', p_network_name='finney', p_netuid=chain, p_limit=1, p_offset=1, p_public_only=True)
    assert store._transport.rpc(RPC, params) == []


@pytest.mark.parametrize('fault', ['request_hash','round_hash','basis_hash','basis_epoch','signature_key','frozen_configuration','uncleared_activation'])
def test_damaged_receipt_fails_closed_without_private_error(state, fault):
    store, c, chain = state
    rid, doc, old = seed(state)
    def mutate(o,n,p):
        if fault == 'basis_hash': o['reward_basis_hash'] = 'sha256:'+'e'*64
        elif fault == 'basis_epoch': o['reward_basis_doc']['effective_reward_epoch'] = 399
        elif fault == 'signature_key': o['reward_basis_doc']['signature']['public_key_hash'] = 'sha256:'+'e'*64
        elif fault == 'frozen_configuration': n['configuration_doc']['private_probe'] = 'modified'
        elif fault == 'uncleared_activation': n['reward_activated_at'] = o['reward_activated_at']
    archive(c, old, mutate=mutate)
    if fault in ('request_hash','round_hash'):
        with c.cursor() as q:
            q.execute('ALTER TABLE lab_arena_scoring_supersessions DISABLE TRIGGER lab_arena_scoring_supersessions_append_only')
            if fault == 'request_hash': q.execute("UPDATE lab_arena_scoring_supersessions SET request_hash='sha256:'||repeat('e',64)WHERE round_id=%s", (rid,))
            else: q.execute("UPDATE lab_arena_scoring_supersessions SET old_round_doc=old_round_doc||'{\"extra_private\":\"not printed\"}' WHERE round_id=%s", (rid,))
            q.execute('ALTER TABLE lab_arena_scoring_supersessions ENABLE TRIGGER lab_arena_scoring_supersessions_append_only')
    with pytest.raises(ArenaStoreError, match='correction_reward_history_(receipt|basis)_invalid') as error:
        read(store, chain)
    assert 'must remain private' not in str(error.value)


def test_rpc_is_service_only_and_migration_replay_preserves_rows(database, state):
    pg, dsn = database
    store, c, chain = state
    _, _, old = seed(state)
    archive(c, old)
    before = read(store, chain)
    with c.cursor() as q:
        q.execute(SQL)
        q.execute(SQL)
        for role in ('anon','authenticated','service_role'):
            q.execute("SELECT has_function_privilege(%s,'public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)','EXECUTE')", (role,))
            assert q.fetchone()[0] is False
        q.execute("SELECT has_function_privilege('lab_arena_service','public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)','EXECUTE')")
        assert q.fetchone()[0] is True
    assert read(store, chain) == before
