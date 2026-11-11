"""Compare final driver progress against the same accepted native SQL evidence.

The existing runner fixture uses deterministic local provider/scorer responses.
No scoring, publication, reward, or weight service method is replaced. Capture
policy is disabled in this fixture; any stored capture rows must remain intact.
"""
from dataclasses import replace
from contextlib import closing
import hashlib
import json
import subprocess
import re
from pathlib import Path

import pytest

from lab_arena import driver, signing
from leadpoet_canonical.arena_weights import verify_accepted_weight_state_signature
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.dynamic_benchmark_lifecycle_test import _harness, _start
from tests.lab_arena.lab_arena_pg_harness import current_service_database, _server_bindir
from tests.lab_arena.test_lab_arena_service_round import keypair


@pytest.fixture
def database(monkeypatch):
    bindir = _server_bindir()
    if bindir is None or not (bindir / 'pg_dump').is_file() or not (bindir / 'psql').is_file():
        pytest.skip('native PostgreSQL snapshot tools are required')
    # Use the same native installation for the server and its dump/restore
    # clients. A newer pg_dump can emit settings an older Docker server
    # cannot restore (for example transaction_timeout in PostgreSQL17).
    monkeypatch.setenv('LAB_ARENA_PG_LOCAL', '1')
    for psycopg, dsn in current_service_database():
        connection = psycopg.connect(**dsn)
        try:
            with connection, connection.cursor() as cursor:
                scripts = Path(__file__).resolve().parents[2] / 'scripts'
                # Match the deployed reward function without the unrelated
                # October29 production-specific data correction (existing681 fixture).
                source = (scripts / '592-arena-2026-10-29-authorized-champion.sql').read_text()
                cursor.execute(re.search(r'DO \$oct29_snapshot592\$.*?\$oct29_snapshot592\$;', source, re.S)[0])
                cursor.execute((scripts / '681-lab-arena-total-emissions-reward-slots.sql').read_text())
        finally:
            connection.close()
        yield psycopg, dsn


def _unsigned(value):
    if isinstance(value, dict):
        return {key: _unsigned(item) for key, item in value.items()
                if key != 'signature'}
    if isinstance(value, list):
        return [_unsigned(item) for item in value]
    return value


def _evidence(connect, round_id):
    with closing(connect()) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' "
                       "AND tablename LIKE 'lab_arena_%' AND "
                       "(tablename LIKE '%training%' OR tablename LIKE '%trajector%' "
                       "OR tablename LIKE '%observation%') ORDER BY tablename")
        tables = ['lab_arena_ledger'] + [row[0] for row in cursor.fetchall()]
        result = {}
        for table in tables:
            # Table names are from pg_catalog in the disposable test database.
            cursor.execute('SELECT to_jsonb(t) FROM public.' + table + ' t')
            result[table] = sorted((row[0] for row in cursor.fetchall()),
                                   key=lambda row: json.dumps(row, sort_keys=True))
        cursor.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r '
                       'WHERE round_id=%s ORDER BY run_id', (round_id,))
        result['runs'] = [{key: value for key, value in row[0].items()
                           if key != 'updated_at'} for row in cursor.fetchall()]
        return result


@pytest.mark.parametrize('baseline_first', [False, True], ids=['two-stage', 'baseline-first'])
def test_same_accepted_sql_evidence_publishes_in_one_tick_with_same_rewards_and_weights(
    database, tmp_path, baseline_first,
):
    psycopg, dsn = database
    connect = lambda: psycopg.connect(**dsn)
    harness = _harness(connect, tmp_path, count=10, margin=0.5,
                       challengers=['Miner'], baseline_first=baseline_first)
    harness.service.config.defaults = replace(harness.service.config.defaults, rewards_enabled=True)
    _start(harness, label='driverfinal', epoch=63200)
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('stage2_scoring', runners=1)
    harness.run_stage_with_runners(1)
    before = _evidence(connect, harness.round_id)
    executions = [run for run in before['runs'] if run['kind'] == 'execute']
    judges = [run for run in before['runs'] if run['kind'] == 'score']
    assert len(executions) == len(judges) == 20
    assert all(run['status'] == 'accepted' for run in executions + judges)
    assert before['lab_arena_ledger']
    assert before['lab_arena_trajectory_events']
    object_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in harness.objects_root.rglob('*') if path.is_file()}
    configuration = harness.service.store.get_round(harness.round_id)['configuration_doc']
    assert not configuration.get('reasoning_capture_policy')

    # Snapshot the same accepted evidence without altering active sessions.
    # The local harness owns this whole disposable PostgreSQL cluster.
    original_config = harness.service.config
    harness.service.store.close()
    admin = psycopg.connect(**{**dsn, 'dbname': 'template1'})
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute('CREATE DATABASE arena_driver_reference TEMPLATE template0')
    finally:
        admin.close()
    reference_dsn = {**dsn, 'dbname': 'arena_driver_reference'}
    bindir = _server_bindir()
    assert bindir is not None
    dumped = subprocess.run(
        [str(bindir / 'pg_dump'), '--dbname', psycopg.extensions.make_dsn(**dsn)],
        check=True, capture_output=True, timeout=30,
    ).stdout
    subprocess.run(
        [str(bindir / 'psql'), '--no-psqlrc', '--set', 'ON_ERROR_STOP=1',
         '--dbname', psycopg.extensions.make_dsn(**reference_dsn)],
        input=dumped, check=True, capture_output=True, timeout=30,
    )
    reference_connect = lambda: psycopg.connect(**reference_dsn)
    stores = [ArenaStore(PsycopgTransport(factory)) for factory in (connect, reference_connect)]
    candidate, reference = [ArenaService(replace(original_config, store=store)) for store in stores]
    try:
        # Same clock, signer, source objects, frozen policy, chain, and DB rows.
        # Original scheduling makes one advance call per tick.
        reference_statuses = []
        for _ in range(3):
            reference.advance_round(harness.round_id)
            reference_statuses.append(reference.store.get_round(harness.round_id)['status'])
        assert reference_statuses == ['stage2_judged', 'scored', 'published']
        reference.activate_pending_rewards()
        summary = driver.drive_once(candidate)
        assert 'failed' not in summary, summary
        assert candidate.store.get_round(harness.round_id)['status'] == 'published'

        rows = [service.store.get_round(harness.round_id) for service in (candidate, reference)]
        assert rows[0]['publication_doc']['final_ranking'] == rows[1]['publication_doc']['final_ranking']
        assert rows[0]['publication_doc']['king_decision'] == rows[1]['publication_doc']['king_decision']
        assert rows[0]['reward_activated_at'] and rows[1]['reward_activated_at'], summary
        assert _unsigned(rows[0]['reward_basis_doc']) == _unsigned(rows[1]['reward_basis_doc'])
        assert _unsigned(rows[0]['publication_doc']) == _unsigned(rows[1]['publication_doc'])
        assert rows[0]['configuration_doc'] == rows[1]['configuration_doc'] == configuration
        after = [_evidence(factory, harness.round_id) for factory in (connect, reference_connect)]
        assert after[0] == after[1]
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
                   for path, digest in object_hashes.items())
        for table in before:
            if table != 'runs':
                assert after[0][table] == before[table]
        # Native scoring may fill score projections but never changes accepted
        # execution/judgment evidence, output refs, claims, or costs.
        for old, new in zip(before['runs'], after[0]['runs']):
            assert {key: value for key, value in new.items() if key not in ('per_icp_score', 'qualification_doc', 'updated_at')} == {
                key: value for key, value in old.items() if key not in ('per_icp_score', 'qualification_doc', 'updated_at')}
        for row in rows:
            assert signing.verify_document_signature(
                row['reward_basis_doc'], hash_field='reward_basis_hash',
                public_key_der=harness.signer.public_key_der,
                expected_public_key_hash=harness.signer.public_key_hash,
            ) == row['reward_basis_doc']['reward_basis_hash']
        epoch = rows[0]['effective_reward_epoch']
        harness.chain.epoch = epoch
        harness.chain.accepted_weight_epoch_scope = lambda: {
            'genesis_hash': '2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03',
            'epoch': epoch, 'valid_from_block': 100, 'valid_until_block': 459,
        }
        states = []
        for service in (candidate, reference):
            service.config.accepted_burn_hotkey = keypair('driver-burn').ss58_address
            state = service.public_weight_state(epoch)['state']
            assert state is not None
            assert verify_accepted_weight_state_signature(
                state, public_key_der=harness.signer.public_key_der,
                expected_public_key_hash=harness.signer.public_key_hash,
            ) == state['state_hash']
            assert service.public_weight_state(epoch)['state'] == state
            states.append(state)
        # ECDSA signatures are randomized. The accepted-state hash includes
        # its nested signed basis, so verify each hash/signature separately
        # above and compare every remaining semantic field here.
        assert {key: value for key, value in _unsigned(states[0]).items() if key != 'state_hash'} == {
            key: value for key, value in _unsigned(states[1]).items() if key != 'state_hash'}
        assert candidate.public_submissions(harness.round_id) == reference.public_submissions(harness.round_id)
    finally:
        for store in stores:
            store.close()
