"""A valid re-signature cannot strand an unchanged finalized observation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading

import pytest

from lab_arena import validator, weight_state
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStoreError
from tests.lab_arena import chain_outcome_retry488_postgres_test as prior
from tests.test_arena_validator import _orchestrator, _protected, _Signer

SQL = Path(__file__).parents[2] / 'scripts/489-lab-arena-chain-outcome-resigning.sql'
PRE = prior.POST
POST = '78af80199274864ccdbe9ee07a820609c1788cd5e29f3865f94a6f8331570190'


def snapshot(database):
    # Hash full receipts so assertion output never contains signatures.
    return hashlib.sha256(json.dumps(prior.rows(database), sort_keys=True, default=str).encode()).hexdigest()


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return prior.NOW


class Signer(_Signer):
    def sign_arena_chain_outcome_v1(self, document):
        return {'signature': '0x' + prior.KEY.sign(weight_state.chain_outcome_message(document).encode()).hex()}


def orchestrator(path, app, state, broadcasts):
    protected = _protected()
    protected.update(epoch=state['epoch'], state_hash=state['state_hash'],
                     weights_hash='2'*64, finalized_block=9287550)
    outcome = dict(status='finalized', finalized=True, finalized_block_hash='4'*64,
                   finalized_block=9287685, commit_included_block=9287550,
                   weights_hash='2'*64, validator_uid=4, last_update=9287550,
                   revealed_weights=[[0, 65535], [1, 100]])
    obj = _orchestrator(path, Signer(protected, [outcome]), broadcasts)
    obj.validator_hotkey = prior.KEY.ss58_address
    obj._verified_state = lambda _epoch, _key: state
    obj.chain.finalized_head = lambda: SimpleNamespace(number=9287550, hash='0x'+'2'*64)
    obj._host_derivation = lambda _state, _hotkeys: dict(state_hash=state['state_hash'],
        weights_hash='2'*64, sparse_uids=[0, 1], sparse_weights_u16=[65535, 100])
    obj.api.submit_chain_outcome = app.record_chain_outcome
    return obj


def same_second_loss(database, path, *, conflict):
    with prior.service(database) as app, pytest.MonkeyPatch.context() as patch:
        patch.setattr(validator, 'datetime', Clock)
        state = prior.accepted(app, 9)
        broadcasts = []
        first = orchestrator(path, app, state, broadcasts)
        assert first.run_once(9) == 'broadcast'
        assert first.run_once(9) == 'finalized'
        canonical = snapshot(database)
        original = validator._read_hashed_json(first.paths.outcome(9))['report_document']
        signed = first.paths.signed(9).read_bytes()
        first.paths.outcome(9).unlink()  # Lose only the disposable local receipt.
        restarted = orchestrator(path, app, state, broadcasts)
        if conflict:
            with pytest.raises(ServiceError, match='chain_outcome_conflict') as error:
                restarted.run_once(9)
            assert error.value.status == 409
        else:
            assert restarted.run_once(9) == 'finalized'
        saved = validator._read_hashed_json(first.paths.outcome(9))
        retry = saved['report_document']
        assert retry['request_id'] == original['request_id']
        assert retry['observed_at'] == original['observed_at']
        assert retry['signature'] != original['signature']
        for document in (original, retry):
            assert prior.verify(document['validator_hotkey'], document['signature'],
                                weight_state.chain_outcome_message(document))
        assert saved['reported'] is (not conflict)
        assert snapshot(database) == canonical
        assert first.paths.signed(9).read_bytes() == signed
        assert restarted.signer.prepares == 0 and broadcasts == ['0xdeadbeef']


@pytest.fixture(scope='module')
def database():
    generator = prior.database.__wrapped__()
    db = next(generator)
    try:
        # Reproduce the actual same-second restart path against merged SQL488.
        with TemporaryDirectory(prefix='arena-outcome489-before-') as path:
            same_second_loss(db, Path(path), conflict=True)
        canonical = snapshot(db)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            before = prior.functions(cur)
            definition = dict((row[0], row[1]) for row in before)[prior.SIGNATURE.removeprefix('public.')]
            assert hashlib.sha256(definition.encode()).hexdigest() == PRE
            text = SQL.read_text()
            for image, error in [(PRE, 'preimage differs'), (POST, 'postimage differs')]:
                with pytest.raises(db[0].Error, match=error):
                    cur.execute(text.replace(image, '0'*64, 1))
                conn.rollback()
                assert prior.functions(cur) == before
            cur.execute('GRANT EXECUTE ON FUNCTION '+prior.SIGNATURE+' TO authenticated')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert prior.functions(cur) == before
            cur.execute('SET ROLE outcome488_migrator')
            cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
            assert cur.fetchone() == (False,)
            cur.execute(text)
            after = prior.functions(cur)
            cur.execute(text)
            assert prior.functions(cur) == after
            cur.execute('RESET ROLE')
            assert [(r[0], r[2:]) for r in before] == [(r[0], r[2:]) for r in after]
            changed = [(a[0], b[1]) for a, b in zip(before, after) if a[1] != b[1]]
            assert len(changed) == 1 and changed[0][0] == prior.SIGNATURE.removeprefix('public.')
            assert hashlib.sha256(changed[0][1].encode()).hexdigest() == POST
            cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_function_privilege('authenticated',%s,'EXECUTE')", (prior.SIGNATURE,))
            assert cur.fetchone() == (False, False)
        assert snapshot(db) == canonical
        yield db
    finally:
        generator.close()


@pytest.fixture
def app(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_chain_outcomes,public.lab_arena_accepted_weight_states')
    with prior.service(database) as value:
        yield value


def test_same_second_normal_restart_keeps_first_receipt(database, app, tmp_path):
    same_second_loss(database, tmp_path, conflict=False)


def test_valid_resignatures_and_old_exact_retry_are_authenticated(database, app, monkeypatch):
    state = prior.accepted(app)
    first, retry = prior.report(state), prior.report(state)
    assert first['request_id'] == retry['request_id'] and first['signature'] != retry['signature']
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    canonical = snapshot(database)
    app.now = lambda: prior.NOW + timedelta(days=30)
    calls = []
    verify = app._config.verify_signature
    app._config.verify_signature = lambda *args: calls.append(True) or verify(*args)
    for document in (first, retry):
        assert app.record_chain_outcome(document) == {'status': 'recorded'}
    assert len(calls) == 2 and snapshot(database) == canonical
    for signature in ('0x'+'00'*64, first['signature']):
        app._config.verify_signature = lambda *args: False
        with pytest.raises(ServiceError, match='chain_outcome_signature_invalid') as error:
            app.record_chain_outcome(dict(first, signature=signature))
        assert error.value.status == 401
    app._config.verify_signature = verify
    monkeypatch.setattr(app._store, 'get_weight_state', lambda *args: None)
    with pytest.raises(ServiceError, match='chain_outcome_state_unknown'):
        app.record_chain_outcome(first)
    assert snapshot(database) == canonical


@pytest.mark.parametrize('change', [
    {'weights_hash': '6'*64}, {'state_hash': 'sha256:'+'5'*64},
    {'finalized_block_hash': '8'*64}, {'finalized_block_number': 9287686},
])
def test_changed_chain_facts_still_conflict(database, app, change):
    state = prior.accepted(app)
    first = prior.report(state)
    app.record_chain_outcome(first)
    canonical = snapshot(database)
    incoming = prior.report(state, **change)
    with pytest.raises(ServiceError) as error:
        app.record_chain_outcome(incoming)
    assert error.value.status == 409
    assert snapshot(database) == canonical


@pytest.mark.parametrize('change', [
    {'schema_version': 'other.v1'}, {'network': 'test'}, {'netuid': 72},
    {'epoch': 25787}, {'validator_hotkey': prior.OTHER.ss58_address},
    {'state_hash': 'sha256:'+'5'*64}, {'weights_hash': '6'*64},
    {'extrinsic_hash': '0x'+'7'*64}, {'finalized_block_hash': '8'*64},
    {'finalized_block_number': 9287686},
])
def test_both_key_chain_fact_controls(database, app, change):
    state = prior.accepted(app)
    first = prior.report(state)
    app.record_chain_outcome(first)
    canonical = snapshot(database)
    incoming = prior.report(state, **change)
    with pytest.raises(ArenaStoreError):
        prior.raw(app._store, incoming, identity=first)
    assert snapshot(database) == canonical


def test_same_second_concurrent_normal_recovery(database, app, tmp_path, monkeypatch):
    monkeypatch.setattr(validator, 'datetime', Clock)
    state = prior.accepted(app, 9)
    broadcasts = []
    first = orchestrator(tmp_path/'one', app, state, broadcasts)
    assert first.run_once(9) == 'broadcast'
    signed = first.paths.signed(9).read_bytes()
    second_path = tmp_path/'two'
    second_path.mkdir()
    validator.ArenaWeightPaths(second_path).signed(9).write_bytes(signed)
    barrier = threading.Barrier(2)
    documents = []

    def recover(path):
        with prior.service(database) as worker:
            record = worker._store.record_chain_outcome
            def racing(**kwargs):
                documents.append(kwargs['outcome_doc'])
                barrier.wait(timeout=5)
                return record(**kwargs)
            worker._store.record_chain_outcome = racing
            obj = orchestrator(path, worker, state, broadcasts)
            assert obj.run_once(9) == 'finalized'
            assert obj.signer.prepares == 0
            assert obj.paths.signed(9).read_bytes() == signed
            assert validator._read_hashed_json(obj.paths.outcome(9))['reported'] is True
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(recover, (tmp_path/'one', second_path)))
    assert len(documents) == 2
    assert documents[0]['request_id'] == documents[1]['request_id']
    assert documents[0]['signature'] != documents[1]['signature']
    stored = prior.rows(database)
    assert len(stored) == 1
    assert any(stored[0]['outcome_doc'] == document for document in documents)
    assert broadcasts == ['0xdeadbeef']
    canonical = snapshot(database)
    for document in documents:
        assert app.record_chain_outcome(document) == {'status': 'recorded'}
    assert snapshot(database) == canonical


def test_crossed_unique_keys_cannot_acknowledge_different_rows(database, app):
    state = prior.accepted(app)
    first = prior.report(state)
    second = prior.report(state, extrinsic_hash='0x'+'7'*64)
    for document in (first, second):
        assert app.record_chain_outcome(document) == {'status': 'recorded'}
    canonical = snapshot(database)
    crossed = dict(second, request_id=first['request_id'])
    with pytest.raises(ArenaStoreError, match='chain_outcome_conflict'):
        prior.raw(app._store, crossed)
    assert snapshot(database) == canonical


def test_concurrent_different_facts_reject_loser(database, app):
    state = prior.accepted(app)
    documents = [prior.report(state, weights_hash=weight*64) for weight in ('2', '6')]
    barrier = threading.Barrier(2)
    def send(document):
        with prior.service(database) as worker:
            record = worker._store.record_chain_outcome
            def racing(**kwargs):
                barrier.wait(timeout=5)
                return record(**kwargs)
            worker._store.record_chain_outcome = racing
            try:
                return worker.record_chain_outcome(document)['status']
            except ServiceError as error:
                assert error.status == 409
                return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(send, documents))
    assert sorted(outcomes) == ['chain_outcome_conflict', 'recorded']
    canonical = snapshot(database)
    for document, outcome in zip(documents, outcomes):
        if outcome == 'recorded':
            assert app.record_chain_outcome(document) == {'status': 'recorded'}
        else:
            with pytest.raises(ServiceError, match='chain_outcome_conflict'):
                app.record_chain_outcome(document)
    assert len(prior.rows(database)) == 1 and snapshot(database) == canonical
