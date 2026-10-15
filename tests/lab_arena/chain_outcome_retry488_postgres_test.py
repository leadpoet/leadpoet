"""Authenticated semantic report retries preserve the first immutable proof."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
from bittensor_wallet import Keypair

from lab_arena import contracts, rewards, signing, validator, weight_state
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/488-lab-arena-chain-outcome-semantic-retry.sql'
SIGNATURE = 'public.lab_arena_record_chain_outcome_v1(text,integer,bigint,text,text,text,jsonb)'
PRE = '4b59587948bfe87087c8cc5de1f8dd68b1ef53932b849387a4082b8e70611c65'
POST = '107dfd90dd15606d3ca169414c8c0376b734d6a813e1e57552b7c7e60a7d65ba'
KEY = Keypair.create_from_uri('//ArenaOutcomeRetry488Validator')
OTHER = Keypair.create_from_uri('//ArenaOutcomeRetry488Independent')
NOW = datetime(2026, 10, 15, 16, 17, tzinfo=timezone.utc)


def verify(hotkey, signature, message):
    try:
        return bool(Keypair(ss58_address=hotkey).verify(message.encode(), bytes.fromhex(signature.removeprefix('0x'))))
    except (ValueError, TypeError):
        return False


@contextmanager
def service(database):
    transport = PsycopgTransport(lambda: database[0].connect(**database[1]))
    value = object.__new__(ArenaService)
    value._store = ArenaStore(transport)
    value._config = SimpleNamespace(network_name='finney', netuid=71, verify_signature=verify)
    value.now = lambda: NOW
    try:
        yield value
    finally:
        transport.close()


def accepted(service, epoch=25786):
    signer = signing.LocalSigner.generate()
    basis = rewards.reward_basis_document(round_id='arena-2026-10-15',
        published_at='2026-10-15T00:00:00Z', finalized_epoch=epoch-1,
        king_outcome='crowned', king_hotkey=KEY.ss58_address)
    basis = signing.sign_document(signer, basis, hash_field='reward_basis_hash')
    document = weight_state.build_accepted_weight_state(signer, network='finney',
        genesis_hash='11'*32, netuid=71, epoch=epoch, valid_from_block=9287500,
        valid_until_block=9287859, reward_basis=basis, burn_hotkey=OTHER.ss58_address,
        issued_at='2026-10-15T00:00:00Z')
    assert service._store.publish_weight_state('finney', 71, epoch, document['state_hash'], document)['status'] == 'accepted'
    return document


def report(state, *, key=KEY, observed_at='2026-10-15T09:01:36Z', **changes):
    body = dict(schema_version=weight_state.CHAIN_OUTCOME_SCHEMA_VERSION,
        network='finney', netuid=71, epoch=state['epoch'], validator_hotkey=key.ss58_address,
        state_hash=state['state_hash'], weights_hash='2'*64, extrinsic_hash='0x'+'3'*64,
        finalized_block_hash='4'*64, finalized_block_number=9287685, observed_at=observed_at)
    body.update(changes)
    request_id = contracts.document_hash(body)
    return dict(body, request_id=request_id,
        signature='0x'+key.sign(('arena_chain_outcome:'+request_id).encode()).hex())


def rows(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_chain_outcomes r ORDER BY network,netuid,epoch,validator_hotkey,request_id')
        return [r[0] for r in cur.fetchall()]


def functions(cur):
    cur.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),proowner,proacl::text,prosecdef,provolatile,proconfig FROM pg_proc WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return cur.fetchall()


def raw(store, document, *, identity=None):
    identity = identity or document
    return store.record_chain_outcome(network=identity['network'], netuid=identity['netuid'],
        epoch=identity['epoch'], validator_hotkey=identity['validator_hotkey'],
        request_id=document['request_id'], extrinsic_hash=identity['extrinsic_hash'], outcome_doc=document)


@pytest.fixture(scope='module')
def database():
    generator = database_with_lab_arena_migration()
    db = next(generator)
    try:
        with service(db) as app:
            state = accepted(app)
            first = report(state)
            retry = report(state, observed_at='2026-10-15T16:08:19Z')
            assert app.record_chain_outcome(first) == {'status': 'recorded'}
            canonical = rows(db)
            # The current service authenticates this valid retry before SQL202
            # rejects its second unique extrinsic key.
            with pytest.raises(ServiceError, match='chain_outcome_conflict') as error:
                app.record_chain_outcome(retry)
            assert error.value.status == 409
            assert rows(db) == canonical
            with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
                before = functions(cur)
                old = {r[0]: r[1] for r in before}
                assert hashlib.sha256(old[SIGNATURE.removeprefix('public.')].encode()).hexdigest() == PRE
                text = SQL.read_text()
                for image, error in [(PRE, 'preimage differs'), (POST, 'postimage differs')]:
                    with pytest.raises(db[0].Error, match=error):
                        cur.execute(text.replace(image, '0'*64, 1))
                    conn.rollback()
                    assert functions(cur) == before
                cur.execute('GRANT EXECUTE ON FUNCTION '+SIGNATURE+' TO authenticated')
                with pytest.raises(db[0].Error, match='security shape differs'):
                    cur.execute(text)
                conn.rollback()
                assert functions(cur) == before
                cur.execute('CREATE ROLE outcome488_migrator NOLOGIN NOSUPERUSER INHERIT')
                cur.execute('GRANT lab_arena_owner TO outcome488_migrator')
                cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO outcome488_migrator WITH GRANT OPTION')
                cur.execute('SET ROLE outcome488_migrator')
                cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
                assert cur.fetchone() == (False,)
                cur.execute(text)
                after = functions(cur)
                cur.execute(text)
                assert functions(cur) == after
                cur.execute('RESET ROLE')
                assert [(r[0], r[2:]) for r in before] == [(r[0], r[2:]) for r in after]
                changed = [(a[0], b[1]) for a, b in zip(before, after) if a[1] != b[1]]
                assert len(changed) == 1 and changed[0][0] == SIGNATURE.removeprefix('public.')
                assert hashlib.sha256(changed[0][1].encode()).hexdigest() == POST
                cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_function_privilege('authenticated',%s,'EXECUTE')", (SIGNATURE,))
                assert cur.fetchone() == (False, False)
            assert app.record_chain_outcome(retry) == {'status': 'recorded'}
            assert rows(db) == canonical
        yield db
    finally:
        generator.close()


@pytest.fixture
def app(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_chain_outcomes,public.lab_arena_accepted_weight_states')
    with service(database) as value:
        yield value


@pytest.mark.parametrize('epoch', [25786, 25792])
def test_exact_and_semantic_reports_preserve_canonical_document_and_time(database, app, epoch):
    state = accepted(app, epoch)
    first = report(state)
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    canonical = rows(database)
    app.now = lambda: datetime(2026, 10, 16, tzinfo=timezone.utc)
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    retry = report(state, observed_at='2026-10-15T16:08:19Z')
    assert retry['request_id'] != first['request_id'] and retry['signature'] != first['signature']
    assert app.record_chain_outcome(retry) == {'status': 'recorded'}
    assert app.public_chain_outcomes(epoch)['outcomes'] == [first]
    assert rows(database) == canonical


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'other.v1'), ('network', 'test'), ('netuid', 72), ('epoch', 25787),
    ('validator_hotkey', OTHER.ss58_address), ('state_hash', 'sha256:'+'5'*64),
    ('weights_hash', '6'*64), ('extrinsic_hash', '0x'+'7'*64),
    ('finalized_block_hash', '8'*64), ('finalized_block_number', 9287686),
])
def test_each_immutable_proof_field_cannot_match_a_colliding_report(database, app, field, value):
    state = accepted(app)
    first = report(state)
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    canonical = rows(database)
    incoming = report(state, observed_at='2026-10-15T16:08:19Z', **{field: value})
    # Bind to the existing key: changed scope columns fail their constraints;
    # every remaining changed proof fails the semantic comparison.
    with pytest.raises(ArenaStoreError):
        raw(app._store, incoming, identity=first)
    assert rows(database) == canonical


@pytest.mark.parametrize('fault', ['signature', 'hash', 'future', 'extra', 'schema', 'network',
                                 'netuid', 'epoch', 'state', 'weights', 'block_hash', 'block_number'])
def test_semantic_retry_still_requires_normal_service_authentication(database, app, fault):
    state = accepted(app)
    first = report(state)
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    canonical = rows(database)
    changes = {'schema': {'schema_version': 'other.v1'}, 'network': {'network': 'test'},
        'netuid': {'netuid': 72}, 'epoch': {'epoch': 25787}, 'state': {'state_hash': 'sha256:'+'5'*64},
        'weights': {'weights_hash': '6'*64}, 'block_hash': {'finalized_block_hash': '8'*64},
        'block_number': {'finalized_block_number': 9287686}}
    incoming = report(state, observed_at='2026-10-15T16:08:19Z', **changes.get(fault, {}))
    if fault == 'signature': incoming['signature'] = '0x'+'00'*64
    elif fault == 'hash': incoming['request_id'] = 'sha256:'+'f'*64
    elif fault == 'future': incoming = report(state, observed_at='2026-10-15T16:23:00Z')
    elif fault == 'extra': incoming['unrecognized'] = True
    with pytest.raises((ServiceError, contracts.ArenaContractError)) as error:
        app.record_chain_outcome(incoming)
    if isinstance(error.value, ServiceError):
        assert error.value.status == (401 if fault == 'signature' else 409)
    assert rows(database) == canonical


def test_same_request_id_retains_exact_document_conflict(database, app):
    state = accepted(app)
    first = report(state)
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    canonical = rows(database)
    incoming = dict(first, signature='0x'+'00'*64)
    with pytest.raises(ServiceError, match='chain_outcome_conflict'):
        app.record_chain_outcome(incoming)
    with pytest.raises(ArenaStoreError, match='chain_outcome_conflict'):
        raw(app._store, incoming)
    assert rows(database) == canonical


def test_independent_validator_extrinsic_and_current_epoch_reports_still_record(database, app):
    state = accepted(app)
    current = accepted(app, 25792)
    documents = [report(state), report(state, key=OTHER),
                 report(state, extrinsic_hash='0x'+'7'*64), report(current)]
    for document in documents:
        assert app.record_chain_outcome(document) == {'status': 'recorded'}
    assert len(rows(database)) == 4


@pytest.mark.parametrize('mode', ['exact', 'equivalent', 'conflicting'])
def test_concurrent_inserts_compare_against_actual_winner(database, app, mode):
    state = accepted(app)
    count = 4 if mode != 'conflicting' else 2
    barrier = threading.Barrier(count)
    documents = [report(state, observed_at=f'2026-10-15T09:01:{36+i:02d}Z',
                        weights_hash=('6'*64 if mode == 'conflicting' and i else '2'*64))
                 for i in range(count)]
    if mode == 'exact': documents = [documents[0]]*count

    def send(document):
        with service(database) as worker:
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

    with ThreadPoolExecutor(max_workers=count) as pool:
        outcomes = list(pool.map(send, documents))
    expected = ['chain_outcome_conflict', 'recorded'] if mode == 'conflicting' else ['recorded']*count
    assert sorted(outcomes) == sorted(expected)
    canonical = rows(database)
    assert len(canonical) == 1 and canonical[0]['outcome_doc'] in documents
    for document, outcome in zip(documents, outcomes):
        if outcome == 'recorded':
            assert app.record_chain_outcome(document) == {'status': 'recorded'}
        else:
            with pytest.raises(ServiceError, match='chain_outcome_conflict'):
                app.record_chain_outcome(document)
    assert rows(database) == canonical


def test_both_unique_keys_cannot_reconcile_two_different_winning_rows(database, app):
    state = accepted(app)
    first = report(state)
    second = report(state, extrinsic_hash='0x'+'7'*64)
    for document in (first, second):
        assert app.record_chain_outcome(document) == {'status': 'recorded'}
    canonical = rows(database)
    crossed = dict(second, request_id=first['request_id'])
    with pytest.raises(ArenaStoreError, match='chain_outcome_conflict'):
        raw(app._store, crossed)
    assert rows(database) == canonical


def test_report_acknowledgment_completes_disposable_historical_recovery(database, app, tmp_path):
    state = accepted(app)
    first = report(state)
    assert app.record_chain_outcome(first) == {'status': 'recorded'}
    canonical = rows(database)
    retried = report(state, observed_at='2026-10-15T16:08:19Z')
    paths = validator.ArenaWeightPaths(tmp_path)
    signed = {'epoch': state['epoch']}
    signed['record_hash'] = contracts.document_hash(signed)
    paths.signed(state['epoch']).write_text(json.dumps(signed))
    record = dict(epoch=state['epoch'], reported=False, report_document=retried)
    record['record_hash'] = contracts.document_hash(record)
    paths.outcome(state['epoch']).write_text(json.dumps(record))
    orchestrator = object.__new__(validator.ArenaWeightOrchestrator)
    orchestrator.paths = paths
    orchestrator.api = SimpleNamespace(submit_chain_outcome=app.record_chain_outcome)
    orchestrator._prior_recovery_failures = set()
    orchestrator._prior_poll_cursor = -1
    orchestrator._operational_logger = None
    assert orchestrator.poll_prior_outcomes(25792)
    saved = validator._read_hashed_json(paths.outcome(state['epoch']))
    assert saved['reported'] is True and saved['report_document'] == retried
    assert paths.signed(state['epoch']).read_text() == json.dumps(signed)
    orchestrator.api.submit_chain_outcome = lambda _: pytest.fail('already reported journal must not POST again')
    assert orchestrator.poll_prior_outcomes(25792)
    assert rows(database) == canonical
