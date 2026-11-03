"""The October 28 one-off changes only the exact, unstarted round policy."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time

import pytest

from lab_arena import contracts
from tests.lab_arena import full_thinking_champion_baseline567_postgres_test as prior
from tests.lab_arena.full_thinking_round_test import approved_registry, opted_in, BASELINE_URL, DIGEST
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/580-arena-2026-10-28-full-thinking-activation.sql'
ROUND = 'arena-2026-10-28'
PRIOR = 'arena-2026-10-27'


@pytest.fixture(scope='module')
def database():
    fixture = prior.database.__wrapped__()
    db = next(fixture)
    try:
        yield db
    finally:
        fixture.close()


@contextmanager
def connection(database):
    conn = database[0].connect(**database[1])
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def configuration():
    service = opted_in()
    service._config.defaults.reasoning_capture_from = '2099-01-01T00:00:00Z'
    config = service.create_round(datetime(2026, 10, 28, tzinfo=timezone.utc))
    config['schedule'] = {key: value.replace('2026-', '2099-')
                          for key, value in config['schedule'].items()}
    config['reasoning_activation_status'] = 'not_ready'
    config['reasoning_activation_reason'] = 'registry_unavailable'
    assert 'reasoning_capture_policy' not in config
    return config


def target(config):
    result = deepcopy(config)
    result.pop('reasoning_activation_status', None)
    result.pop('reasoning_activation_reason', None)
    result.update(reasoning_capture_policy='full_generated_thinking_v1',
                  training_minimum_icp_coverage_percent=80,
                  training_maximum_missing_reasoning_percent=1,
                  reasoning_route_registry=approved_registry(),
                  reasoning_baseline_source_sha256=DIGEST,
                  baseline_source_url=BASELINE_URL,
                  execution_icp_cap_microusd=6_000_000,
                  cost_per_company_microusd=6_000_000,
                  execution_cap_microusd=6_000_000 * contracts.benchmark_icp_count(config))
    return result


def migration(cursor, config):
    source = SQL if SQL.exists() else SQL.with_suffix('.sql.template')
    text = source.read_text()
    for name, value in (
        ('expected_before', configuration_hash(cursor, config)),
        ('expected_after', configuration_hash(cursor, target(config))),
        ('baseline_url', BASELINE_URL), ('baseline_hash', DIGEST),
    ):
        text, count = re.subn(rf"({name} CONSTANT TEXT := )'[^']*'", rf"\g<1>'{value}'", text)
        assert count == 1
    text, count = re.subn(r"(registry CONSTANT JSONB := )'.*?'::JSONB;",
                         lambda match: match[1] + "'" + json.dumps(approved_registry()) + "'::JSONB;",
                         text, flags=re.DOTALL)
    assert count == 1
    return text


def configuration_hash(cursor, config):
    cursor.execute("SELECT encode(extensions.digest((%s::jsonb)::text,'sha256'),'hex')", (json.dumps(config),))
    return cursor.fetchone()[0]


def seed(cursor, config=None):
    config = config or configuration()
    cursor.execute('SET session_replication_role=replica')
    cursor.execute('DELETE FROM public.lab_arena_runs WHERE round_id IN (%s,%s)', (ROUND, PRIOR))
    cursor.execute('DELETE FROM public.lab_arena_submissions WHERE round_id IN (%s,%s)', (ROUND, PRIOR))
    cursor.execute('DELETE FROM public.lab_arena_rounds WHERE round_id IN (%s,%s)', (ROUND, PRIOR))
    for round_id in (PRIOR, ROUND):
        doc = dict(config, round_id=round_id)
        cursor.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,rewards_enabled) "
                       "VALUES(%s,'open',%s::jsonb,true)", (round_id, json.dumps(doc)))
    # Existing accepted sources remain pending under the new round policy.
    for suffix in ('pending', 'second'):
        cursor.execute("INSERT INTO public.lab_arena_submissions "
                       "(submission_id,round_id,miner_hotkey,status,submission_doc) "
                       "VALUES(%s,%s,%s,'accepted',%s::jsonb)",
                       ('activation580-' + suffix, ROUND, hotkey('activation580-' + suffix),
                        json.dumps({'source_ref': suffix + '.tar.gz', 'preserve': True})))
    cursor.execute('SET session_replication_role=origin')
    return config


def snapshot(cursor):
    result = {}
    for name, order in (('lab_arena_rounds', 'round_id'), ('lab_arena_submissions', 'submission_id'),
                        ('lab_arena_runs', 'run_id')):
        cursor.execute(f'SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY {order}),\'[]\'::jsonb) '
                       f'FROM public.{name} x WHERE round_id IN (%s,%s)', (ROUND, PRIOR))
        result[name] = cursor.fetchone()[0]
    cursor.execute("SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),"
                   "pg_get_functiondef(t.tgfoid),to_jsonb(p)-'prosrc') ORDER BY t.tgname) "
                   "FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid "
                   "WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal")
    result['triggers'] = cursor.fetchone()[0]
    return result


def test_activation_preserves_oct27_and_admitted_sources_and_restores_guard(database):
    with connection(database) as conn:
        with conn.cursor() as cur:
            config = seed(cur)
            text = migration(cur, config)
            before = snapshot(cur)
            # Reproduce the write-once barrier before the guarded activation.
            with pytest.raises(database[0].Error, match='configuration is write-once'):
                cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',
                            (json.dumps(target(config)), ROUND))
            cur.execute(text)
            after = snapshot(cur)
            assert after['lab_arena_submissions'] == before['lab_arena_submissions']
            assert after['lab_arena_runs'] == before['lab_arena_runs']
            assert after['triggers'] == before['triggers']
            assert after['lab_arena_rounds'][0] == before['lab_arena_rounds'][0]
            changed = after['lab_arena_rounds'][1]
            assert changed['configuration_doc'] == target(config)
            assert contracts.validate_round_configuration(changed['configuration_doc']) == target(config)
            assert {k: v for k, v in changed.items() if k not in ('configuration_doc', 'updated_at')} == {
                k: v for k, v in before['lab_arena_rounds'][1].items() if k not in ('configuration_doc', 'updated_at')}
            cur.execute(text)
            assert snapshot(cur) == after
            # The exception does not survive as a writable service setting.
            cur.execute("SELECT set_config('lab_arena.full_thinking_baseline_freeze_round',%s,false)", (ROUND,))
            with pytest.raises(database[0].Error, match='configuration is write-once'):
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                            "jsonb_set(configuration_doc,'{training_minimum_icp_coverage_percent}','79') WHERE round_id=%s",
                            (ROUND,))


@pytest.mark.parametrize('fault', ['stale_config', 'cutoff', 'review_window', 'participants', 'benchmark', 'run_started',
                                  'frozen', 'king', 'reviewing', 'reviewed', 'funding', 'published', 'reward',
                                  'generation', 'confirmation', 'stage3', 'fallback'])
def test_activation_rejects_stale_or_started_state_without_changes(database, fault):
    with connection(database) as conn:
        with conn.cursor() as cur:
            config = configuration()
            if fault == 'cutoff':
                config['schedule']['submission_cutoff'] = '2000-01-01T00:00:00Z'
            elif fault == 'review_window':
                config['schedule']['submission_cutoff'] = (
                    datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
            seed(cur, config)
            text = migration(cur, config)
            cur.execute('SET session_replication_role=replica')
            if fault == 'stale_config':
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc||'{\"stale\":true}' WHERE round_id=%s", (ROUND,))
            elif fault in ('participants', 'benchmark', 'funding', 'published', 'reward',
                           'generation', 'confirmation', 'stage3', 'fallback'):
                expression = {'participants': "participants='[]'::jsonb", 'benchmark': "benchmark_ref='started.json'",
                              'funding': 'champion_funding_frozen=true', 'published': "status='published'",
                              'reward': "reward_basis_hash='sha256:" + 'a' * 64 + "'",
                              'generation': 'stage_generation=1',
                              'confirmation': "confirmation_bank_ref='confirmation-started.json',confirmation_bank_hash='sha256:" + 'c' * 64 + "'",
                              'stage3': "stage3_scoring_plan_doc='{}'::jsonb",
                              'fallback': "champion_fallback_providers=ARRAY['openrouter']::text[]"}[fault]
                cur.execute('UPDATE public.lab_arena_rounds SET ' + expression + ' WHERE round_id=%s', (ROUND,))
            elif fault in ('frozen', 'king', 'reviewing', 'reviewed'):
                expression = {'frozen': "status='frozen'", 'king': 'is_king=true',
                              'reviewed': "code_review_status='passed',code_review_claim='sha256:" + 'b' * 64 +
                              "',code_review_doc='{}'::jsonb,code_review_attempts=1,code_review_started_at=clock_timestamp(),"
                              "code_review_expires_at=NULL",
                              'reviewing': "code_review_status='reviewing',code_review_claim='sha256:" + 'a' * 64 +
                              "',code_review_doc=NULL,code_review_attempts=1,code_review_started_at=clock_timestamp(),"
                              "code_review_expires_at=clock_timestamp()+interval '1 hour'"}[fault]
                cur.execute("UPDATE public.lab_arena_submissions SET " + expression + " WHERE submission_id='activation580-pending'")
            elif fault == 'run_started':
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,scoring_generation_floor) "
                            "VALUES('activation580-run','activation580-assignment',%s,'activation580-pending',%s,1,0,1,'pending',0)",
                            (ROUND, hotkey('activation580-pending')))
            cur.execute('SET session_replication_role=origin')
            before = snapshot(cur)
            with pytest.raises(database[0].Error, match='exact state or intake window differs'):
                cur.execute(text)
            cur.execute('ROLLBACK')
            assert snapshot(cur) == before


def test_wrong_target_hash_and_guard_drift_fail_closed(database):
    with connection(database) as conn:
        with conn.cursor() as cur:
            config = seed(cur)
            text = migration(cur, config)
            before = snapshot(cur)
            wrong = re.sub(r"(expected_after CONSTANT TEXT := )'[0-9a-f]{64}'", r"\g<1>'" + '0' * 64 + "'", text)
            with pytest.raises(database[0].Error, match='target configuration differs'):
                cur.execute(wrong)
            cur.execute('ROLLBACK')
            assert snapshot(cur) == before
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_rounds_write_once_v1()'::regprocedure)")
            definition = cur.fetchone()[0]
            try:
                cur.execute(definition.replace('BEGIN\n', 'BEGIN\n  NULL;\n', 1))
                drifted = snapshot(cur)
                with pytest.raises(database[0].Error, match='round guard differs'):
                    cur.execute(text)
                cur.execute('ROLLBACK')
                assert snapshot(cur) == drifted
            finally:
                cur.execute(definition)


def test_failed_readback_rolls_back_update_and_trigger_disable(database):
    with connection(database) as conn, conn.cursor() as cur:
        config = seed(cur)
        text = migration(cur, config)
        # A future/concurrent trigger that changes protected state must make the
        # whole activation fail, including restoration of the original guard.
        cur.execute("CREATE FUNCTION public.activation580_corrupt_readback() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$BEGIN NEW.benchmark_ref := 'unexpected.json'; RETURN NEW; END$$")
        cur.execute('CREATE TRIGGER zz_activation580_corrupt_readback BEFORE UPDATE '
                    'ON public.lab_arena_rounds FOR EACH ROW EXECUTE FUNCTION public.activation580_corrupt_readback()')
        try:
            before = snapshot(cur)
            with pytest.raises(database[0].Error, match='changed protected state'):
                cur.execute(text)
            cur.execute('ROLLBACK')
            assert snapshot(cur) == before
        finally:
            cur.execute('DROP TRIGGER zz_activation580_corrupt_readback ON public.lab_arena_rounds')
            cur.execute('DROP FUNCTION public.activation580_corrupt_readback()')


def test_admission_cannot_race_the_locked_configuration_change(database):
    with connection(database) as conn, conn.cursor() as cur:
        config = seed(cur)
        # Pause immediately before the write, after the real migration locks
        # and snapshots. This is the only test-only SQL substitution.
        text = migration(cur, config).replace(
            '  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER',
            '  PERFORM pg_advisory_xact_lock(580, 28);\n'
            '  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER', 1)
        cur.execute('SELECT pg_advisory_lock(580, 28)')

        def activate():
            with connection(database) as worker, worker.cursor() as cursor:
                cursor.execute(text)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(activate)
            try:
                deadline = time.monotonic() + 5
                while True:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' "
                                'AND classid=580 AND objid=28 AND NOT granted)')
                    if cur.fetchone()[0]:
                        break
                    if future.done():
                        future.result()
                    assert time.monotonic() < deadline, 'activation did not reach locked write'
                    time.sleep(0.01)
                with connection(database) as entrant, entrant.cursor() as cursor:
                    cursor.execute("SET lock_timeout='100ms'")
                    with pytest.raises(database[0].Error, match='lock timeout'):
                        cursor.execute("UPDATE public.lab_arena_submissions SET updated_at=clock_timestamp() "
                                       "WHERE submission_id='activation580-pending'")
            finally:
                cur.execute('SELECT pg_advisory_unlock(580, 28)')
            future.result(timeout=10)
        cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (ROUND,))
        assert cur.fetchone()[0] == target(config)


def _assert_historical_registry_preserved(current_registry, registry):
    assert current_registry['schema_version'] == registry['schema_version']
    assert current_registry['baseline_route'] == registry['baseline_route']
    historical_routes = registry['approved_routes']
    assert current_registry['approved_routes'][:len(historical_routes)] == historical_routes


@pytest.mark.parametrize('damage', [None, 'row_evidence', 'order', 'baseline'])
def test_historical_registry_allows_appended_routes_but_rejects_historical_changes(damage):
    from lab_arena import reasoning_capture

    original = reasoning_capture.load_registry()
    current = deepcopy(original)
    added = deepcopy(original['approved_routes'][0])
    added['model_id'] = added['model_version'] = 'example/future-reviewed-model'
    current['approved_routes'].append(added)
    if damage == 'row_evidence':
        current['approved_routes'][0]['evidence']['probe_manifest_sha256'] = '0' * 64
    elif damage == 'order':
        current['approved_routes'][:2] = current['approved_routes'][1::-1]
    elif damage == 'baseline':
        current['baseline_route']['api'] = 'openrouter.chat'
    if damage is None:
        _assert_historical_registry_preserved(current, original)
    else:
        with pytest.raises(AssertionError):
            _assert_historical_registry_preserved(current, original)


def test_committed_activation_contains_complete_valid_source_and_route_pins():
    from lab_arena import reasoning_capture

    assert SQL.exists(), 'release must contain the exact .sql, not a template'
    text = SQL.read_text()
    assert '__BEFORE_' not in text and '__AFTER_' not in text
    for name in ('expected_before', 'expected_after'):
        assert re.search(rf"{name} CONSTANT TEXT := '[0-9a-f]{{64}}';", text)
    assert re.search(r"baseline_url CONSTANT TEXT := 'https://github[.]com/leadpoet/leadpoet-sales-agent/archive/[0-9a-f]{40}[.]tar[.]gz';", text)
    assert re.search(r"baseline_hash CONSTANT TEXT := 'sha256:[0-9a-f]{64}';", text)
    encoded = re.search(r"registry CONSTANT JSONB := '((?:''|[^'])*)'::JSONB;", text)
    assert encoded is not None
    registry = reasoning_capture.validate_registry(json.loads(encoded[1].replace("''", "'")))
    current_registry = reasoning_capture.load_registry()
    # This migration is an immutable historical activation. Later tested
    # routes can be appended without rewriting its frozen registry or defaults.
    _assert_historical_registry_preserved(current_registry, registry)
    assert registry['approved_routes'] and registry['baseline_route'] is not None
