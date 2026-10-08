from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import httpx
import pytest
from fastapi.testclient import TestClient
from lab_arena.public_dashboard import history_snapshot
from lab_arena.store import ArenaStore, PostgrestTransport, PsycopgTransport
from lab_arena.service import ServiceError
from lab_arena.api import create_app


def fixture(count=30):
    rows, calls = [], []
    for index in range(count, 0, -1):
        day = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat()
        participants = [{"submission_id": f"day{index}-miner{i}", "miner_hotkey": f"5Miner{i}", "is_baseline": i == 0} for i in range(3)]
        rows.append({"round_id": f"round-{index}", "status": "published", "created_at": day, "evaluation_date": day[:10],
                     "configuration_doc": {}, "publication_doc": {"participants": participants,
                     "final_ranking": [{"submission_id": p["submission_id"], "final_score": i} for i, p in enumerate(participants)]},
                     "private_field": "must-not-leak"})
    def list_rounds(**kwargs):
        calls.append(kwargs)
        assert (kwargs['status'], kwargs['mode'], kwargs['network_name'], kwargs['netuid']) == ('published', 'live', 'finney', 71)
        assert kwargs['history_order'] is True
        selected = [row for row in rows if (not kwargs.get('evaluation_date') or row['evaluation_date'] == kwargs['evaluation_date'])
                    and (not kwargs.get('before_round') or (row['created_at'], row['round_id']) < kwargs['before_round'])]
        return selected[:kwargs['limit']]
    service = SimpleNamespace(_config=SimpleNamespace(mode='live'), _chain_scope=lambda: ('finney', 71),
                              now=lambda: datetime(2026, 10, 8, tzinfo=timezone.utc),
                              _store=SimpleNamespace(list_rounds=list_rounds, list_runs=lambda *_a, **_k: [],
                                                     list_runtime_starts=lambda *_a, **_k: []),
                              _round=lambda rid: next(row for row in rows if row['round_id'] == rid))
    return service, calls, rows


def test_history_pages_preserve_zero_and_filter_across_rounds():
    service, calls, rows = fixture()
    pages, cursor = [], None
    while True:
        page = history_snapshot(service, cursor=cursor)
        pages += page['submissions']
        cursor = page['next_cursor']
        if cursor is None: break
    assert len(pages) == 90
    assert len({s['submission_id'] for s in pages}) == 90
    assert pages[0]['final_score'] == 0
    miner = history_snapshot(service, hotkey='miner2')
    assert len(miner['submissions']) == 25 and all(s['miner_hotkey'] == '5Miner2' for s in miner['submissions'])
    day = history_snapshot(service, hotkey='Miner1', day='2026-01-03')
    assert len(day['submissions']) == 1 and day['submissions'][0]['submission_id'] == 'day2-miner1'
    assert calls[-1]['evaluation_date'] == '2026-01-03'
    assert 'must-not-leak' not in str(miner)


def test_sparse_search_has_bounded_queries_and_continuation():
    service, calls, rows = fixture(120)
    rows[-1]['publication_doc']['participants'][1]['miner_hotkey'] = 'RareMiner'
    page = history_snapshot(service, hotkey='RareMiner')
    assert len(calls) == 4 and not page['submissions'] and page['next_cursor']
    page = history_snapshot(service, hotkey='RareMiner', cursor=page['next_cursor'])
    assert len(page['submissions']) == 1 and page['next_cursor'] is None


def test_newer_publication_does_not_shift_continuation():
    service, calls, rows = fixture()
    page = history_snapshot(service)
    rows.insert(0, {**rows[0], 'round_id': 'new', 'created_at': '2026-03-01T00:00:00+00:00'})
    second = history_snapshot(service, cursor=page['next_cursor'])
    assert len(second['submissions']) == 25
    assert not {s['submission_id'] for s in page['submissions']} & {s['submission_id'] for s in second['submissions']}


def test_pinned_history_and_invalid_cursor():
    service, calls, rows = fixture()
    service._config.pinned_round_id = 'round-2'
    result = history_snapshot(service)
    assert len(result['submissions']) == 3 and not calls
    with pytest.raises(ServiceError):
        history_snapshot(service, cursor='bad')


def test_day_filter_reaches_database_as_bound_equality():
    calls = []
    store = ArenaStore(SimpleNamespace(select=lambda *args, **kwargs: calls.append((args, kwargs)) or []))
    store.list_rounds(evaluation_date='2026-09-02', status='published', mode='live', network_name='finney', netuid=71)
    assert calls[0][1]['filters']['evaluation_date'] == '2026-09-02'


def test_history_route_rejects_invalid_dates_before_query():
    calls = []
    service = SimpleNamespace(public_history=lambda **kwargs: calls.append(kwargs) or {'submissions': [], 'rounds': [], 'next_cursor': None})
    with TestClient(create_app(service)) as client:
        for day in ['2026-99-99', '2026-02-30', 'invalid']:
            assert client.get('/arena/v1/history', params={'day': day}).status_code == 422
        assert not calls
        result = client.get('/arena/v1/history', params={'day': '2026-10-07'})
        assert result.status_code == 200 and result.headers['cache-control'] == 'no-store'
        assert calls[0]['day'] == '2026-10-07'


def test_keyset_transport_queries_are_bounded_and_parameterized():
    before = ('2026-09-02T01:02:03.123456+00:00', 'arena-2026-09-02')
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=[]))) as http:
        transport = PostgrestTransport('https://example.test', service_key='sb_secret_test', http_client=http)
        ArenaStore(transport).list_rounds(status='published', limit=25, before_round=before, history_order=True)
    params = requests[0].url.params
    assert params['order'] == 'created_at.desc,round_id.desc'
    assert params['limit'] == '25'
    assert params['or'] == '(created_at.lt.2026-09-02T01:02:03.123456+00:00,and(created_at.eq.2026-09-02T01:02:03.123456+00:00,round_id.lt.arena-2026-09-02))'
    queries = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
    transport = object.__new__(PsycopgTransport)
    transport._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    transport._release = lambda connection: None
    ArenaStore(transport).list_rounds(status='published', limit=25, before_round=before, history_order=True)
    sql, values = queries[0]
    assert '(created_at, round_id) < (%s, %s)' in sql
    assert 'ORDER BY created_at DESC, round_id DESC LIMIT 25' in sql
    assert values == ['published', *before]


def test_cursor_rejects_changed_filters():
    service, calls, rows = fixture()
    cursor = history_snapshot(service, hotkey='5Miner')['next_cursor']
    for filters in ({'hotkey': 'Different'}, {'hotkey': '5Miner', 'day': '2026-01-03'}):
        with pytest.raises(ServiceError) as exc:
            history_snapshot(service, cursor=cursor, **filters)
        assert exc.value.code == 'history_cursor_invalid'


def test_history_code_versions_use_stored_round_runs_and_only_returned_submission_ids():
    service, _calls, rows = fixture(2)
    reads = []
    hotkey = '5' + 'A' * 47
    # One published submission failed, while zero remains a real completed score.
    rows[0]['publication_doc']['final_ranking'][1]['final_score'] = None

    def list_runs(round_id, **kwargs):
        reads.append(('runs', round_id, kwargs))
        assert kwargs['submission_ids'] == [f"day{round_id[-1]}-miner1"]
        return [{'run_id': f'{round_id}-execution', 'assignment_id': 'private-assignment',
                 'submission_id': kwargs['submission_ids'][0], 'kind': 'execute',
                 'status': 'accepted', 'runner_hotkey': hotkey, 'attempt': 1,
                 'source_commit': ('a' if round_id == 'round-2' else 'b') * 40,
                 'source_dirty': 'clean', 'private_payload': 'must-not-leak'}]

    def list_starts(round_id, **kwargs):
        reads.append(('starts', round_id, kwargs))
        raise AssertionError('Known completed source metadata must not read any trajectory')

    service._store.list_runs = list_runs
    service._store.list_runtime_starts = list_starts
    page = history_snapshot(service, hotkey='Miner1', limit=2)
    assert len(page['submissions']) == 2 and len(reads) == 2
    newer, older = page['submissions']
    assert newer['status'] == 'scoring_failed'
    assert newer['evaluation']['state'] == 'failed'
    assert older['evaluation']['state'] == 'completed'
    assert newer['evaluation']['code_versions'][0]['commit'] == 'a' * 40
    assert older['evaluation']['code_versions'][0]['commit'] == 'b' * 40
    assert all(item['evaluation']['validators'] == [] for item in page['submissions'])
    for private in ['private-assignment', 'private_payload', 'round-2-execution', 'must-not-leak']:
        assert private not in str(page)
