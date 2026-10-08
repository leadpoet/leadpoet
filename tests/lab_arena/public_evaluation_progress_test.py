from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from lab_arena import public_dashboard

NOW = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
PRIMARY = '5' + 'A' * 47
AUDITOR = '5' + 'B' * 47


def run(**overrides):
    return {
        'kind': 'score', 'status': 'leased', 'runner_hotkey': PRIMARY,
        'lease_expires_at': '2026-10-08T13:00:00Z', 'submission_id': 'miner',
        **overrides,
    }


def test_progress_distinguishes_queue_execution_scoring_and_finalization():
    project = lambda runs: public_dashboard.evaluation_progress(runs, NOW)
    assert project([]) == {'state': 'queued', 'validators': []}
    assert project([run(status='pending')]) == {'state': 'queued', 'validators': []}
    assert project([run(status='accepted')]) == {'state': 'finalizing', 'validators': []}
    assert project([run(kind='execute'), run(runner_hotkey=AUDITOR), run()]) == {
        'state': 'evaluating', 'validators': [
            {'hotkey': PRIMARY, 'phase': 'executing'},
            {'hotkey': PRIMARY, 'phase': 'scoring'},
            {'hotkey': AUDITOR, 'phase': 'scoring'},
        ],
    }
    assert project([run(status='pending'), run()])['state'] == 'evaluating'
    for expiry in [None, 'invalid', '2026-10-08T12:00:00Z', '2026-10-08T11:59:59Z']:
        assert project([run(lease_expires_at=expiry)]) == {'state': 'queued', 'validators': []}
    assert project([run(runner_hotkey='')]) == {'state': 'unavailable', 'validators': []}


def test_snapshot_uses_one_allowlisted_read_and_never_leaks_private_run_fields():
    row = {'round_id': 'round', 'status': 'stage2', 'configuration_doc': {}}
    records = [{'submission_id': sid, 'miner_hotkey': PRIMARY, 'status': 'frozen'}
               for sid in ['miner', 'waiting', 'done', 'not-public']]
    calls = []

    def list_runs(round_id, **kwargs):
        calls.append((round_id, kwargs))
        return [run(source_ref='secret-source', lease_token_hash='secret-lease',
                    output_doc={'private': 'private-output'}, per_icp_score=99),
                run(submission_id='not-public', runner_hotkey=AUDITOR)]

    service = SimpleNamespace(_round=lambda _: row, now=lambda: NOW,
        _store=SimpleNamespace(list_runs=list_runs, list_submissions=lambda *_a, **_k: records))
    with patch.object(public_dashboard, '_participants', return_value=[
        {'submission_id': sid} for sid in ['miner', 'waiting', 'done']
    ]), patch.object(public_dashboard, '_stage1_scores', return_value={}), \
         patch.object(public_dashboard, '_completed_scores', return_value={'done': {'final_score': 0}}), \
         patch.object(public_dashboard.source_disclosure, 'disclosure_status', return_value={'available': False}):
        result = public_dashboard.submissions_snapshot(service, 'round')
    by_id = {item['submission_id']: item for item in result['submissions']}
    assert by_id['miner']['status'] == 'scoring'
    assert by_id['miner']['evaluation']['state'] == 'evaluating'
    assert by_id['waiting']['evaluation'] == {'state': 'queued', 'validators': []}
    assert by_id['done']['final_score'] == 0
    assert 'evaluation' not in by_id['done']
    assert 'not-public' not in by_id
    assert calls == [('round', {'columns': 'run_id,submission_id,kind,status,runner_hotkey,lease_expires_at'})]
    for secret in ['secret-source', 'secret-lease', 'private-output', AUDITOR, '99']:
        assert secret not in str(result)


def test_compact_progress_read_keeps_pagination_and_excludes_private_columns():
    from lab_arena.store import ArenaStore
    columns = 'run_id,submission_id,kind,status,runner_hotkey,lease_expires_at'
    calls = []

    def select(table, **kwargs):
        calls.append((table, kwargs))
        assert kwargs['columns'] == columns
        assert kwargs['filters'] == {'round_id': 'round'}
        assert kwargs['order'] == 'run_id'
        assert kwargs['limit'] == 500
        start = 0 if kwargs['after_run_id'] is None else 500
        return [{'run_id': f'run-{index:04d}'} for index in range(start, min(start + 500, 501))]

    rows = ArenaStore(SimpleNamespace(select=select)).list_runs('round', columns=columns)
    assert len(rows) == 501
    assert len(calls) == 2
    assert calls[1][1]['after_run_id'] == 'run-0499'
