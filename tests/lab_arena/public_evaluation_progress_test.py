"""Public Arena progress must use saved assignment and source identities."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
import json

import httpx
import pytest
from lab_arena import public_dashboard, runner, trajectory
from lab_arena.store import ArenaStore, PostgrestTransport, PsycopgTransport

NOW = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
PRIMARY = '5' + 'A' * 47
AUDITOR = '5' + 'B' * 47


def run(**overrides):
    return {
        'run_id': 'run:1', 'assignment_id': 'assignment', 'attempt': 1,
        'lease_generation': 1, 'kind': 'score', 'status': 'leased',
        'runner_hotkey': PRIMARY, 'lease_expires_at': '2026-10-08T13:00:00Z',
        'submission_id': 'miner', **overrides,
    }


def start(row, **overrides):
    return {key: row[key] for key in ('run_id', 'assignment_id', 'attempt', 'runner_hotkey', 'submission_id')} | {
        'start_lease_generation': str(row['lease_generation']),
        'source_commit': 'a' * 40, 'source_dirty': 'clean', **overrides,
    }


def test_progress_distinguishes_queue_execution_scoring_and_finalization():
    project = lambda runs: public_dashboard.evaluation_progress(runs, NOW)
    assert project([])['state'] == 'queued'
    assert project([run(status='pending')])['counts']['queued'] == 1
    assert project([run(status='accepted')])['state'] == 'finalizing'
    result = project([run(kind='execute', assignment_id='execute'),
                      run(runner_hotkey=AUDITOR, assignment_id='judge-auditor'), run()])
    assert result['state'] == 'evaluating'
    assert result['active_count'] == 3
    assert result['validators'] == [
        {'hotkey': PRIMARY, 'phase': 'executing', 'commit': None, 'working_tree': 'unknown'},
        {'hotkey': PRIMARY, 'phase': 'scoring', 'commit': None, 'working_tree': 'unknown'},
        {'hotkey': AUDITOR, 'phase': 'scoring', 'commit': None, 'working_tree': 'unknown'},
    ]
    assert project([run(status='pending', assignment_id='waiting'), run()])['state'] == 'evaluating'
    for expiry in [None, 'invalid', '2026-10-08T12:00:00Z', '2026-10-08T11:59:59Z']:
        result = project([run(lease_expires_at=expiry)])
        assert result['state'] == 'queued' and not result['validators']
    assert project([run(runner_hotkey='')])['state'] == 'unavailable'


def test_retries_count_latest_assignment_and_preserve_each_recorded_code_version():
    failed = run(status='failed', source_commit='a' * 40, source_dirty='dirty')
    retry = run(run_id='run:2', attempt=2, status='pending', runner_hotkey=None)
    result = public_dashboard.evaluation_progress([failed, retry], NOW)
    assert result['state'] == 'retrying'
    assert result['counts'] == {'queued': 0, 'active': 0, 'completed': 0, 'failed': 0, 'retrying': 1}
    complete = run(run_id='run:2', attempt=2, status='accepted', source_commit='b' * 40,
                   source_dirty='clean', runner_hotkey=AUDITOR)
    result = public_dashboard.evaluation_progress([failed, complete], NOW, outcome='completed')
    assert result['state'] == 'completed' and result['counts']['completed'] == 1
    assert [item['commit'] for item in result['code_versions']] == ['a' * 40, 'b' * 40]
    assert not result['validators'] and result['active_count'] == 0
    # An accepted assignment remains the effective authority even if a later
    # failed row is present. No lifecycle or accepted score changes are made.
    assert public_dashboard.evaluation_progress([complete, run(attempt=3, status='failed')], NOW)['counts']['completed'] == 1
    assert public_dashboard.evaluation_progress([failed], NOW)['state'] == 'failed'


@pytest.mark.parametrize('overrides', [
    {'start_lease_generation': '1'}, {'start_lease_generation': None},
    {'start_lease_generation': True}, {'runner_hotkey': AUDITOR},
    {'run_id': 'other-run'}, {'assignment_id': 'other-assignment'}, {'attempt': 2},
])
def test_active_commit_never_uses_stale_same_run_or_other_assignment_start(overrides):
    row = run(lease_generation=2)
    result = public_dashboard.evaluation_progress([row], NOW, runtime_starts=[start(row, **overrides)])
    assert result['validators'][0]['commit'] is None
    assert result['code_versions'][0]['commit'] is None


def test_active_commit_uses_exact_generation_and_rejects_conflicting_receipts():
    row = run(lease_generation=2, source_commit='c' * 40, source_dirty='dirty')
    stale = start(row, start_lease_generation='1', source_commit='b' * 40)
    current = start(row)
    result = public_dashboard.evaluation_progress([row], NOW, runtime_starts=[stale, current])
    assert result['validators'][0]['commit'] == 'a' * 40
    assert result['validators'][0]['working_tree'] == 'clean'
    conflict = start(row, source_commit='d' * 40)
    assert public_dashboard.evaluation_progress([row], NOW, runtime_starts=[current, conflict])['validators'][0]['commit'] is None


def test_historical_result_source_never_uses_present_runtime_or_heartbeat():
    historical = run(status='accepted', source_commit='a' * 40, source_dirty='dirty')
    result = public_dashboard.evaluation_progress([historical], NOW, outcome='completed',
                                                runtime_starts=[start(historical, source_commit='b' * 40)])
    assert result['code_versions'][0]['commit'] == 'a' * 40
    unknown = run(status='accepted', source_commit='secret-invalid-url', lease_generation=2)
    result = public_dashboard.evaluation_progress([unknown], NOW, outcome='failed', runtime_starts=[start(unknown, start_lease_generation='1')])
    assert result['state'] == 'failed' and result['code_versions'][0]['commit'] is None
    assert 'secret-invalid-url' not in json.dumps(result)


def test_runtime_start_generation_survives_actual_trajectory_sanitization():
    content = runner._runtime_started_content({'lease_generation': 7})
    receipt = trajectory.event('runtime.started', content, occurred_at=NOW)
    assert receipt['content']['lease_generation'] == 7
    assert receipt['content']['validator_source_commit'] == content['validator_source_commit']
    for value in [None, True, 0, '7']:
        assert 'lease_generation' not in runner._runtime_started_content({'lease_generation': value})


def test_snapshot_uses_two_compact_reads_scoped_to_public_rows_and_never_leaks_private_data():
    row = {'round_id': 'round', 'status': 'stage2', 'configuration_doc': {}}
    records = [{'submission_id': sid, 'miner_hotkey': PRIMARY, 'status': 'frozen'}
               for sid in ['miner', 'waiting', 'done', 'not-public']]
    calls = []

    def list_runs(round_id, **kwargs):
        calls.append(('runs', round_id, kwargs))
        return [run(source_ref='secret-source', lease_token_hash='secret-lease',
                    output_doc={'private': 'private-output'}, per_icp_score=99),
                run(submission_id='not-public', runner_hotkey=AUDITOR)]

    def list_starts(round_id, **kwargs):
        calls.append(('starts', round_id, kwargs))
        return [start(run(), provider_response='private-provider'), start(run(submission_id='not-public', runner_hotkey=AUDITOR))]

    service = SimpleNamespace(_round=lambda _: row, now=lambda: NOW,
        _store=SimpleNamespace(list_runs=list_runs, list_runtime_starts=list_starts,
                               list_submissions=lambda *_a, **_k: records))
    with patch.object(public_dashboard, '_participants', return_value=[
        {'submission_id': sid} for sid in ['miner', 'waiting', 'done']
    ]), patch.object(public_dashboard, '_stage1_scores', return_value={}), \
         patch.object(public_dashboard, '_completed_scores', return_value={'done': {'final_score': 0}}), \
         patch.object(public_dashboard.source_disclosure, 'disclosure_status', return_value={'available': False}):
        result = public_dashboard.submissions_snapshot(service, 'round')
    by_id = {item['submission_id']: item for item in result['submissions']}
    assert by_id['miner']['evaluation']['state'] == 'evaluating'
    assert by_id['miner']['evaluation']['validators'][0]['commit'] == 'a' * 40
    assert by_id['waiting']['evaluation']['state'] == 'queued'
    assert by_id['done']['final_score'] == 0
    assert by_id['done']['evaluation']['state'] == 'completed'
    assert 'not-public' not in by_id
    assert calls == [('runs', 'round', {'columns': public_dashboard._EVALUATION_COLUMNS,
                                      'submission_ids': ['done', 'miner', 'waiting']}),
                     ('starts', 'round', {'run_ids': ['run:1']})]
    for secret in ['secret-source', 'secret-lease', 'private-output', 'private-provider', AUDITOR, '99', 'assignment', 'run:1']:
        assert secret not in json.dumps(result)


def test_compact_reads_keep_pagination_and_exclude_private_columns():
    calls = []

    def select(table, **kwargs):
        calls.append((table, kwargs))
        assert kwargs['limit'] == 500
        if table == 'lab_arena_runs':
            assert kwargs['submission_ids'] == ['miner']
            assert kwargs['columns'] == public_dashboard._EVALUATION_COLUMNS
            assert kwargs['filters'] == {'round_id': 'round'}
            assert kwargs['order'] == 'run_id'
            start = 0 if kwargs['after_run_id'] is None else 500
            return [{'run_id': f'run-{index:04d}'} for index in range(start, min(start + 500, 501))]
        assert kwargs['run_ids'] == ['run:1']
        assert kwargs['filters'] == {'round_id': 'round', 'event_kind': 'runtime.started'}
        assert kwargs['order'] == 'trajectory_id'
        assert 'content,' not in kwargs['columns'] and 'provider' not in kwargs['columns']
        return [{'trajectory_id': index} for index in range(kwargs['offset'], min(kwargs['offset'] + 500, 501))]

    store = ArenaStore(SimpleNamespace(select=select))
    assert len(store.list_runs('round', columns=public_dashboard._EVALUATION_COLUMNS, submission_ids=['miner'])) == 501
    assert len(store.list_runtime_starts('round', run_ids=['run:1'])) == 501
    assert len(calls) == 4 and calls[1][1]['after_run_id'] == 'run-0499'
    assert calls[3][1]['offset'] == 500


def test_compact_runtime_projection_has_matching_postgrest_and_bound_sql():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=[]))) as http:
        store = ArenaStore(PostgrestTransport('https://example.test', service_key='sb_secret_test', http_client=http))
        store.list_runs('round', columns=public_dashboard._EVALUATION_COLUMNS, submission_ids=['miner'])
        store.list_runtime_starts('round', run_ids=['run:1'])
    assert requests[0].url.params['submission_id'] == 'in.(miner)'
    assert requests[0].url.params['select'] == public_dashboard._EVALUATION_COLUMNS
    assert 'terminal_cause' in requests[0].url.params['select'].split(',')
    assert requests[1].url.params['event_kind'] == 'eq.runtime.started'
    queries = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
    transport = object.__new__(PsycopgTransport)
    transport._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    transport._release = lambda _: None
    ArenaStore(transport).list_runs('round', columns=public_dashboard._EVALUATION_COLUMNS, submission_ids=['miner'])
    ArenaStore(transport).list_runtime_starts('round', run_ids=['run:1'])
    assert "result_doc #>> '{resource_summary,validator_source_commit}' AS source_commit" in queries[0][0]
    assert 'terminal_cause' in queries[0][0] and 'terminal_doc' not in queries[0][0]
    assert "content ->> 'lease_generation' AS start_lease_generation" in queries[1][0]
    assert 'submission_id = ANY(%s)' in queries[0][0]
    assert queries[0][1] == ['round', ['miner']]
    assert 'run_id = ANY(%s)' in queries[1][0]
    assert queries[1][1] == ['round', 'runtime.started', ['run:1']]
    assert requests[1].url.params['run_id'] == 'in.(run:1)'


def test_terminal_failure_wins_mixed_progress_and_same_attempt_reclaims_are_retries():
    accepted = run(status='accepted', assignment_id='accepted')
    failed = run(status='failed', assignment_id='failed')
    result = public_dashboard.evaluation_progress([accepted, failed], NOW)
    assert result['state'] == 'failed'
    assert result['counts']['completed'] == result['counts']['failed'] == 1
    # An already accepted public score is still complete, including zero.
    assert public_dashboard.evaluation_progress([accepted, failed], NOW, outcome='completed')['state'] == 'completed'
    reclaimed = run(status='pending', attempt=1, lease_generation=2)
    assert public_dashboard.evaluation_progress([reclaimed], NOW)['state'] == 'retrying'
    submitted = run(status='submitted', assignment_id='submitted')
    assert public_dashboard.evaluation_progress([failed, submitted], NOW)['state'] == 'finalizing'


def test_runtime_reads_use_small_indexed_batches_and_reject_filter_syntax():
    calls = []
    store = ArenaStore(SimpleNamespace(select=lambda table, **kwargs: calls.append(kwargs) or []))
    ids = [f'run.{index}:1' for index in range(51)]
    assert store.list_runtime_starts('round', run_ids=ids) == []
    assert [len(call['run_ids']) for call in calls] == [25, 25, 1]
    assert sorted(value for call in calls for value in call['run_ids']) == sorted(ids)
    from lab_arena.store import ArenaStoreError
    with httpx.Client(transport=httpx.MockTransport(lambda _request: pytest.fail('Invalid ID reached HTTP'))) as http:
        transport = PostgrestTransport('https://example.test', service_key='sb_secret_test', http_client=http)
        for ids in [[], ['bad,(run)'], ['bad\nrun'], ['x' * 201]]:
            with pytest.raises(ArenaStoreError):
                transport.select('lab_arena_trajectory_events', run_ids=ids)


@pytest.mark.parametrize("identity", [
    {"runner_hotkey": None, "lease_generation": 0},
    {"runner_hotkey": "invalid", "lease_generation": 1},
    {"runner_hotkey": PRIMARY, "lease_generation": 0},
    {"runner_hotkey": PRIMARY, "lease_generation": None},
])
def test_never_started_failed_assignments_do_not_query_trajectory(identity):
    rows = [run(status="failed", **identity)]
    reads = []
    def list_runs(_round_id, **kwargs):
        reads.append(kwargs)
        return rows
    def list_starts(*_args, **_kwargs):
        pytest.fail("A failed run without a runtime identity cannot use a start receipt")
    service = SimpleNamespace(now=lambda: NOW,
        _store=SimpleNamespace(list_runs=list_runs, list_runtime_starts=list_starts))
    entries = [{"submission_id": "miner", "status": "scoring_failed"}]
    public_dashboard._attach_evaluations(service, "round", entries)
    assert len(reads) == 1
    assert entries[0]["evaluation"]["state"] == "failed"
    assert entries[0]["evaluation"]["counts"]["failed"] == 1


def test_unclaimed_stage_closed_assignments_explain_all_49_incomplete_models():
    reads = []
    entries = [{"submission_id": f"miner-{index}", "status": "scoring"} for index in range(49)]
    rows = [run(
        run_id=f"run:{index}:{icp}", assignment_id=f"assignment:{index}:{icp}",
        submission_id=f"miner-{index}", kind="execute", status="failed",
        terminal_cause="stage_closed", runner_hotkey=None, lease_generation=0,
        result_doc=None, terminal_doc={"error": "private-host-error"},
    ) for index in range(49) for icp in range(10)]
    def list_runs(_round_id, **kwargs):
        reads.append(kwargs)
        return rows
    service = SimpleNamespace(now=lambda: NOW, _store=SimpleNamespace(
        list_runs=list_runs,
        list_runtime_starts=lambda *_a, **_k: pytest.fail("Unclaimed runs have no start receipt"),
    ))
    public_dashboard._attach_evaluations(service, "round", entries)
    assert len(reads) == 1 and "terminal_cause" in reads[0]["columns"].split(",")
    for entry in entries:
        evaluation = entry["evaluation"]
        assert evaluation["state"] == "failed"
        assert evaluation["failure_reasons"] == ["execution_window"]
        assert evaluation["counts"] == {"queued": 0, "active": 0, "completed": 0, "failed": 10, "retrying": 0}
        assert not evaluation["validators"] and not evaluation["code_versions"]
    assert "private-host-error" not in json.dumps(entries)


@pytest.mark.parametrize("cause,reason", [
    ("credential_error", "provider_credentials"), ("stage_closed", "execution_window"),
    ("judge_error", "review"), ("judge_timeout", "review"), ("provider_error", "provider"),
    ("model_timeout", "execution"), ("invalid_output", "execution"),
    ("budget_exhausted", "execution"), ("model_error", "execution"),
    ("lease_expired", "unknown"), ("worker_lost", "unknown"), ("result_rejected", "unknown"),
    (None, "unknown"), ("private-unrecognized-error", "unknown"), ({"error": "private"}, "unknown"),
])
def test_failure_reasons_publish_only_fixed_categories(cause, reason):
    projected = public_dashboard.evaluation_progress([
        run(status="failed", terminal_cause=cause, terminal_doc={"error": "private-detail"}),
    ], NOW)
    assert projected["failure_reasons"] == [reason]
    assert "private" not in json.dumps(projected)


def test_failure_reasons_follow_effective_assignments_and_do_not_change_retry_or_scores():
    failed = run(status="failed", terminal_cause="credential_error")
    for successor in [run(attempt=2, status="pending", runner_hotkey=None),
                      run(attempt=2, status="accepted"),
                      run(attempt=0, status="accepted")]:
        projected = public_dashboard.evaluation_progress([failed, successor], NOW)
        assert "failure_reasons" not in projected
        assert projected["counts"]["failed"] == 0
    latest = run(attempt=2, status="failed", terminal_cause="judge_timeout")
    assert public_dashboard.evaluation_progress([failed, latest], NOW)["failure_reasons"] == ["review"]
    mixed = [failed, run(assignment_id="other", status="failed", terminal_cause="stage_closed")]
    projected = public_dashboard.evaluation_progress(mixed, NOW)
    assert projected["failure_reasons"] == ["execution_window", "provider_credentials"]
    assert "failure_reasons" not in public_dashboard.evaluation_progress(mixed, NOW, outcome="completed")
