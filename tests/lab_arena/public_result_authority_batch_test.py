"""Exact public authority batching keeps validators and request-local bounds."""
from copy import deepcopy
from math import ceil
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import company_judgments, contracts, integrity, signal_judgments
from lab_arena.service import ArenaService, ServiceError, _PublicResultEvidenceReads
from lab_arena.store import ArenaStore, ArenaStoreError, ArenaStoreUnavailable, PostgrestTransport, PsycopgTransport
from tests.lab_arena.company_judgments_test import _miss
from tests.lab_arena.legacy_company_judgments_test import _input
from tests.lab_arena.public_result_company_evidence_reads_test import (
    _make_current_score_the_authority, _public_company_result,
)
from tests.lab_arena.scoring_validator_attribution_test import (
    ROUND_ID, SUBMISSION_ID, VALIDATOR_A, _execution, _score,
)


class Transport:
    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def select(self, table, **kwargs):
        self.calls.append((table, kwargs))
        rows = deepcopy(self.tables.get(table, []))
        if kwargs.get('authority_pairs') is not None:
            rows = [row for row in rows if (row['cache_key'], row['authority_slot']) in kwargs['authority_pairs']]
        if kwargs.get('run_ids') is not None:
            rows = [row for row in rows if row['run_id'] in kwargs['run_ids']]
        for key, value in (kwargs.get('filters') or {}).items():
            rows = [row for row in rows if row.get(key) == value]
        if kwargs.get('order') == 'signal_use_identity':
            rows.sort(key=lambda row: (row['run_id'], row['company_index'], row['signal_index']))
        offset = kwargs.get('offset') or 0
        return rows[offset:offset + kwargs.get('limit', len(rows))]


def _scaling_case(count):
    executions, scores, companies, signals, uses = [], [], [], [], []
    for index in range(count):
        document = _input(('Company %d' % index,))
        ref = company_judgments.build_company_scopes(
            scoring_input=document, round_id=ROUND_ID, network_name='finney', netuid=71,
            scorer_image_digest='sha256:' + 'a' * 64,
            scorer_image_reference='registry/scorer@sha256:' + 'a' * 64,
            integrity_policy=integrity.POLICY, company_judgment_policy=company_judgments.POLICY,
        )[0]
        execution = _execution('execute-%03d' % index, index)
        score = _score('score-%03d' % index, execution, VALIDATOR_A,
            company_judgment_refs=[ref], claim_response={'company_judgment_cache': {
                'schema_version': company_judgments.LEASE_SCHEMA_VERSION,
                'hits': [], 'misses': [_miss(ref)],
            }})
        executions.append(execution)
        scores.append(score)
        companies.append({'cache_key': ref['cache_key'], 'authority_slot': 0})
        signal_key = 'sha256:%064x' % index
        signals.append({'cache_key': signal_key, 'authority_slot': index % 2})
        uses.append({'run_id': score['run_id'], 'company_index': 0, 'signal_index': 0,
                     'cache_key': signal_key, 'authority_slot': index % 2})
    transport = Transport({'lab_arena_runs': executions + scores,
                           'lab_arena_company_judgments': companies,
                           'lab_arena_signal_judgments': signals,
                           'lab_arena_signal_judgment_uses': uses})
    service = ArenaService.__new__(ArenaService)
    service._store = ArenaStore(transport)
    context = {'round_id': ROUND_ID, 'configuration_doc': {
        'scorer_policy': {'signal_judgment_policy': signal_judgments.POLICY}}}
    rows = {row['run_id']: row for row in executions + scores}
    return service, transport, executions, scores, companies, signals, context, rows


def _read_case(count, *, batch):
    service, transport, executions, scores, companies, signals, context, rows = _scaling_case(count)
    reads = _PublicResultEvidenceReads()
    if batch:
        service._prefetch_public_result_evidence(
            reads, round_id=ROUND_ID, execution_runs=executions, score_runs=scores,
            public_run_ids={row['run_id'] for row in executions}, source_rows=rows,
            round_context=context,
        )
    result = []
    for company, signal, score in zip(companies, signals, scores):
        result.append((reads.company(service._store, company['cache_key'], company['authority_slot']),
                       reads.signal(service._store, signal['cache_key'], signal['authority_slot']),
                       reads.uses(service._store, score['run_id']),
                       reads.run(service._store, score['run_id'])))
    return result, transport.calls


@pytest.mark.parametrize('count', [1, 25, 26, 51])
def test_query_count_scales_by_bounded_batches_with_exact_equal_raw_reads(count):
    expected, prior = _read_case(count, batch=False)
    actual, calls = _read_case(count, batch=True)
    assert actual == expected
    assert len(prior) == 4 * count
    assert len(calls) == 3 * ceil(count / 25)
    assert all(len(kwargs.get('authority_pairs', kwargs.get('run_ids', []))) <= 25
               for _, kwargs in calls)


@pytest.mark.parametrize('reused', [False, True])
@pytest.mark.parametrize('published', [False, True])
def test_complete_public_response_bytes_match_without_batch_helpers(reused, published):
    service, ref, _, cached, _, objects, counts = _public_company_result()
    if not reused:
        _make_current_score_the_authority(service, ref, cached, objects)
    if not published:
        row = service._round(ROUND_ID)
        row['status'] = 'stage2'
        row['participants'] = row['publication_doc']['participants']
        service.completed_submission_scores = lambda _row: {
            SUBMISSION_ID: {'execution_runs': [service._store.rows['execute-current']],
                            'final_score': 1.0}}
    expected = service.public_results(ROUND_ID, SUBMISSION_ID)
    transport = Transport({'lab_arena_company_judgments': [{**cached, 'authority_slot': 0}]})
    service._store.get_company_judgment_batch = ArenaStore(transport).get_company_judgment_batch
    service._store.get_runs = lambda ids: {key: service._store.rows[key] for key in ids
                                          if key in service._store.rows}
    counts.clear()
    actual = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert counts['company_rows'] == counts['run_rows'] == 0
    assert len(transport.calls) == 1


@pytest.mark.parametrize('fault', ['missing', 'foreign', 'bad_hash', 'bad_source', 'bad_scope'])
def test_batch_rows_still_pass_every_company_authority_validator(fault):
    service, ref, _, cached, source, _, _ = _public_company_result()
    row = {**deepcopy(cached), 'authority_slot': 0}
    if fault == 'bad_hash': row['evidence_hash'] = 'sha256:' + 'f' * 64
    elif fault == 'bad_source': source['claim_request_hash'] = 'sha256:' + 'f' * 64
    elif fault == 'bad_scope': row['scope_doc']['round_id'] = 'foreign'
    elif fault == 'foreign': row['authority_slot'] = 1
    transport = Transport({'lab_arena_company_judgments': [] if fault == 'missing' else [row]})
    service._store.get_company_judgment_batch = ArenaStore(transport).get_company_judgment_batch
    with pytest.raises(ServiceError, match='public_company_verification_unavailable'):
        service.public_results(ROUND_ID, SUBMISSION_ID)


@pytest.mark.parametrize('failure', [ArenaStoreError, ArenaStoreUnavailable])
def test_batch_errors_are_not_silently_replaced_by_single_reads(failure):
    service, _, _, _, _, _, counts = _public_company_result()
    def fail(_pairs): raise failure('bounded read failure')
    service._store.get_company_judgment_batch = fail
    with pytest.raises(ArenaStoreUnavailable if failure is ArenaStoreUnavailable else ServiceError):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    assert counts['company_rows'] == 0


def test_prefetch_excludes_private_failed_foreign_and_superseded_scores():
    service, transport, executions, scores, _, _, context, rows = _scaling_case(5)
    scores[1]['status'] = 'failed'
    scores[2]['round_id'] = 'foreign'
    scores[3]['stage_generation'] = 0
    newer = {**scores[3], 'run_id': 'new-score', 'stage_generation': 1,
             'company_judgment_refs': None}
    service._prefetch_public_result_evidence(
        _PublicResultEvidenceReads(), round_id=ROUND_ID, execution_runs=executions,
        score_runs=scores + [newer], public_run_ids={run['run_id'] for run in executions[:4]},
        source_rows=rows, round_context=context,
    )
    company_calls = [kwargs['authority_pairs'] for table, kwargs in transport.calls
                     if table == 'lab_arena_company_judgments']
    assert company_calls == [[(scores[0]['company_judgment_refs'][0]['cache_key'], 0)]]
    use_calls = [kwargs['run_ids'] for table, kwargs in transport.calls
                 if table == 'lab_arena_signal_judgment_uses']
    assert use_calls == [[scores[0]['run_id']]]


@pytest.mark.parametrize('table', ['lab_arena_company_judgments', 'lab_arena_signal_judgments'])
def test_authority_pair_filters_have_exact_parameterized_encodings(table):
    pairs = [('sha256:' + 'a' * 64, 0), ('sha256:' + 'b' * 64, 3)]
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=[]))) as http:
        PostgrestTransport('https://example.test', service_key='sb_secret_test', http_client=http).select(
            table, authority_pairs=pairs, limit=26)
    params = requests[0].url.params
    assert params['or'] == '(and(cache_key.eq.%s,authority_slot.eq.0),and(cache_key.eq.%s,authority_slot.eq.3))' % (pairs[0][0], pairs[1][0])
    assert 'cache_key' not in params and 'authority_slot' not in params
    queries = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _connection: None
    direct.select(table, authority_pairs=pairs, limit=26)
    sql, values = queries[0]
    assert '((cache_key = %s AND authority_slot = %s) OR (cache_key = %s AND authority_slot = %s))' in sql
    assert values == [pairs[0][0], 0, pairs[1][0], 3]
    assert pairs[0][0] not in sql


@pytest.mark.parametrize('kind', ['postgrest', 'psycopg'])
@pytest.mark.parametrize('overrides', [
    {'authority_pairs': []}, {'authority_pairs': [('key', True)]},
    {'authority_pairs': [('key', -1)]}, {'authority_pairs': [('key', 1.0)]},
    {'authority_pairs': [(None, 0)]}, {'authority_pairs': [('bad,key', 0)]},
    {'authority_pairs': [('key', 0)] * 2},
    {'authority_pairs': [('key', index) for index in range(26)]},
    {'authority_pairs': [('key', 0)], 'filters': {'authority_slot': 1}},
    {'authority_pairs': [('key', 0)], 'cache_keys': ['key']},
])
def test_invalid_pair_filters_fail_before_either_transport_performs_io(kind, overrides):
    if kind == 'postgrest':
        client = httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail('invalid pair reached HTTP')))
        transport = PostgrestTransport('https://example.test', service_key='sb_secret_test', http_client=client)
    else:
        transport = PsycopgTransport(lambda: pytest.fail('invalid pair opened database'), role=None)
    try:
        with pytest.raises(ArenaStoreError):
            transport.select('lab_arena_company_judgments', **overrides)
    finally:
        transport.close()


@pytest.mark.parametrize('pairs', [[('key', 1), ('key', True)], [('key', True), ('key', 1)],
                                   [('key', 0), (None, 0)], [('key', 0), ('key', False)]])
def test_store_validates_each_identity_before_deduplication(pairs):
    transport = Transport({})
    with pytest.raises(ArenaStoreError):
        ArenaStore(transport).get_company_judgment_batch(pairs)
    assert transport.calls == []


@pytest.mark.parametrize('rows', [
    [{'cache_key': 'other', 'authority_slot': 0}],
    [{'cache_key': 'key', 'authority_slot': True}],
    [{'cache_key': 'key', 'authority_slot': 0}] * 2,
    [None],
])
def test_store_rejects_foreign_duplicate_or_malformed_batch_rows(rows):
    transport = SimpleNamespace(select=lambda *_args, **_kwargs: rows)
    with pytest.raises(ArenaStoreError):
        ArenaStore(transport).get_company_judgment_batch([('key', 0)])


def test_missing_variant_never_aliases_another_slot():
    transport = Transport({'lab_arena_company_judgments': [
        {'cache_key': 'key', 'authority_slot': 1}]})
    assert ArenaStore(transport).get_company_judgment_batch([('key', 0)]) == {}


@pytest.mark.parametrize('bound', ['MAX_IDENTITIES', 'MAX_BYTES'])
def test_seeding_obeys_cache_bounds_and_does_not_cache_failed_validation(bound):
    service, transport, executions, scores, companies, _, context, rows = _scaling_case(2)
    reads = _PublicResultEvidenceReads()
    setattr(reads, bound, 0)
    service._prefetch_public_result_evidence(
        reads, round_id=ROUND_ID, execution_runs=executions, score_runs=scores,
        public_run_ids={row['run_id'] for row in executions}, source_rows=rows,
        round_context=context,
    )
    assert reads.runs == reads.companies == reads.signals == reads.signal_uses == {}
    assert reads._bytes == 0
    reads.company(service._store, companies[0]['cache_key'], 0)
    assert 'filters' in transport.calls[-1][1]


def test_new_request_observes_updated_batch_authority_and_sources():
    service, _, _, cached, _, _, _ = _public_company_result()
    transport = Transport({'lab_arena_company_judgments': [{**cached, 'authority_slot': 0}]})
    service._store.get_company_judgment_batch = ArenaStore(transport).get_company_judgment_batch
    first = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert first['scoring_attribution']['unattributed_icp_count'] == 0
    transport.tables['lab_arena_company_judgments'][0]['evidence_hash'] = 'sha256:' + 'f' * 64
    with pytest.raises(ServiceError, match='public_company_verification_unavailable'):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    assert len(transport.calls) == 2


def _use_rows(count):
    return [{'run_id': 'run-%d' % (index // 500), 'company_index': (index % 500) // 100,
             'signal_index': index % 100} for index in range(count)]


def test_signal_use_batch_pages_completely_and_keeps_missing_runs_empty():
    rows = _use_rows(501)
    transport = Transport({'lab_arena_signal_judgment_uses': rows})
    got = ArenaStore(transport).get_signal_judgment_uses_batch(['run-0', 'run-1', 'missing'])
    assert got == {'run-0': rows[:500], 'run-1': rows[500:], 'missing': []}
    assert [kwargs['offset'] for _, kwargs in transport.calls] == [0, 500]
    assert all(kwargs['order'] == 'signal_use_identity' and kwargs['limit'] == 500
               for _, kwargs in transport.calls)


@pytest.mark.parametrize('fault', ['duplicate', 'foreign', 'bool_company', 'bad_signal', 'unordered', 'overlong'])
def test_signal_use_batch_rejects_invalid_or_truncated_identity_sequences(fault):
    rows = _use_rows(2)
    if fault == 'duplicate': rows[1] = deepcopy(rows[0])
    elif fault == 'foreign': rows[1]['run_id'] = 'foreign'
    elif fault == 'bool_company': rows[1]['company_index'] = False
    elif fault == 'bad_signal': rows[1]['signal_index'] = 100
    elif fault == 'unordered': rows.reverse()
    else: rows = _use_rows(501)
    store = ArenaStore(SimpleNamespace(select=lambda *_args, **_kwargs: rows))
    with pytest.raises(ArenaStoreError):
        store.get_signal_judgment_uses_batch(['run-0'])


def test_signal_use_batch_detects_a_repeated_page_at_schema_bound():
    rows = _use_rows(500)
    store = ArenaStore(SimpleNamespace(select=lambda *_args, **_kwargs: rows))
    with pytest.raises(ArenaStoreError, match='order or uniqueness'):
        store.get_signal_judgment_uses_batch(['run-0'])


def _signal_public_case(*, batched, fault=None):
    from tests.lab_arena.training_export_signal_test import _signal_fixture
    from tests.lab_arena.scoring_validator_attribution_test import VALIDATOR_B
    service, ref, _, cached, source, _, _ = _public_company_result()
    source['company_judgment_refs'] = [ref]
    service._round(ROUND_ID)['configuration_doc']['scorer_policy']['signal_judgment_policy'] = signal_judgments.POLICY
    row = deepcopy(_signal_fixture()['signal_judgment_rows'][0])
    scope = row['scope_doc']
    scope['round_id'] = ROUND_ID
    scope['cache_key'] = contracts.document_hash({key: value for key, value in scope.items() if key != 'cache_key'})
    row.update(cache_key=scope['cache_key'], source_score_run_id='signal-original',
               source_scored_run_id='signal-execute', source_runner_hotkey=VALIDATOR_B,
               runner_authority_exclusions=[VALIDATOR_B])
    signal_execution = _execution('signal-execute', 0, submission_id='signal-submission')
    signal_score = _score('signal-original', signal_execution, VALIDATOR_B,
                         claim_request_hash=row['source_claim_request_hash'],
                         lease_generation=row['source_lease_generation'])
    service._store.rows.update({run['run_id']: run for run in (signal_execution, signal_score)})
    use = {key: row[key] for key in ('cache_key', 'authority_slot', 'signal_input_hash',
                                   'source_score_run_id', 'judgment_hash')}
    use.update(run_id='score-source', company_index=0, signal_index=0)
    if fault == 'hash': row['judgment_hash'] = 'sha256:' + 'f' * 64
    elif fault == 'scope': scope['round_id'] = 'foreign'
    elif fault == 'foreign_source': signal_score['round_id'] = 'foreign'
    elif fault == 'claim': signal_score['claim_request_hash'] = 'sha256:' + 'f' * 64
    elif fault == 'judgment':
        row['judgment_doc'] = {}
        row['judgment_hash'] = use['judgment_hash'] = contracts.document_hash({})
    transport = Transport({
        'lab_arena_company_judgments': [{**cached, 'authority_slot': 0}],
        'lab_arena_signal_judgments': [] if fault == 'missing' else [row],
        'lab_arena_signal_judgment_uses': [use],
    })
    reader = ArenaStore(transport)
    service._store.signal_authority_exclusions = lambda _: [VALIDATOR_A]
    service._store.get_signal_judgment_uses = reader.get_signal_judgment_uses
    service._store.get_signal_judgment = reader.get_signal_judgment
    if batched:
        service._store.get_runs = lambda ids: {key: service._store.rows[key] for key in ids
                                              if key in service._store.rows}
        service._store.get_company_judgment_batch = reader.get_company_judgment_batch
        service._store.get_signal_judgment_batch = reader.get_signal_judgment_batch
        service._store.get_signal_judgment_uses_batch = reader.get_signal_judgment_uses_batch
    return service.public_results(ROUND_ID, SUBMISSION_ID), transport.calls


@pytest.mark.parametrize('fault', [None, 'hash', 'scope', 'foreign_source', 'claim', 'judgment', 'missing'])
def test_batched_signal_attribution_matches_full_public_response_and_rejects_invalid_authority(fault):
    expected, _ = _signal_public_case(batched=False, fault=fault)
    actual, calls = _signal_public_case(batched=True, fault=fault)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert actual['scoring_attribution']['unattributed_icp_count'] == (0 if fault is None else 1)
    assert len(calls) == 3
    assert all('filters' not in kwargs for _, kwargs in calls)


def test_signal_use_transport_filters_and_unique_order_match_in_both_backends():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=[]))) as http:
        store = ArenaStore(PostgrestTransport('https://example.test', service_key='sb_secret_test', http_client=http))
        assert store.get_signal_judgment_uses_batch(['run-2', 'run-1']) == {'run-1': [], 'run-2': []}
    params = requests[0].url.params
    assert params['run_id'] == 'in.(run-1,run-2)'
    assert params['order'] == 'run_id.asc,company_index.asc,signal_index.asc'
    assert params['limit'] == '500' and params['offset'] == '0'
    queries = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _connection: None
    ArenaStore(direct).get_signal_judgment_uses_batch(['run-2', 'run-1'])
    sql, values = queries[0]
    assert 'run_id = ANY(%s)' in sql
    assert 'ORDER BY run_id ASC, company_index ASC, signal_index ASC LIMIT 500 OFFSET 0' in sql
    assert values == [['run-1', 'run-2']]


@pytest.mark.parametrize('table', ['lab_arena_runs', 'lab_arena_judgment_cache'])
def test_authority_filter_refuses_other_tables(table):
    direct = PsycopgTransport(lambda: pytest.fail('foreign table opened database'), role=None)
    try:
        with pytest.raises(ArenaStoreError, match='authority pair filter is invalid'):
            direct.select(table, authority_pairs=[('key', 0)])
    finally:
        direct.close()


def test_valid_hashed_company_evidence_still_excludes_the_recipient_authority():
    outcomes = []
    for batched in (False, True):
        service, _, lease, cached, _, _, _ = _public_company_result()
        evidence_hash = cached['evidence_hash']
        assert contracts.document_hash(cached['evidence_doc']) == evidence_hash
        assert VALIDATOR_A in cached['evidence_doc']['runner_authority_exclusions']
        # The source remains valid. This recipient is explicitly excluded from
        # consuming that source, with no change to evidence or lease hashes.
        service._store.rows['score-current']['miner_hotkey'] = VALIDATOR_A
        if batched:
            transport = Transport({'lab_arena_company_judgments': [{**cached, 'authority_slot': 0}]})
            service._store.get_company_judgment_batch = ArenaStore(transport).get_company_judgment_batch
        with pytest.raises(ServiceError, match='public_company_verification_unavailable') as error:
            service.public_results(ROUND_ID, SUBMISSION_ID)
        outcomes.append((str(error.value), error.value.status))
        assert cached['evidence_hash'] == lease['hits'][0]['evidence_hash'] == evidence_hash
        assert contracts.document_hash(cached['evidence_doc']) == evidence_hash
    assert outcomes[0] == outcomes[1]
