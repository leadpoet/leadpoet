"""The service rejects late replacements before it performs credential work."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStoreError

CUTOFF = datetime(2026, 9, 15, 23, tzinfo=timezone.utc)
ROUND = {
    'round_id': 'arena-2026-09-16', 'status': 'open',
    'configuration_doc': {'schedule': {
        'submission_open': '2026-09-15T00:00:00Z',
        'submission_cutoff': '2026-09-16T00:00:00Z',
    }},
}


def _service(moment, *, replacement=True):
    row = {
        'submission_id': 'sub-new', 'round_id': ROUND['round_id'],
        'miner_hotkey': '5' + 'A' * 47, 'status': 'uploading',
        'source_ref': 'arena/arena-2026-09-16/sources/sub-new.tar.gz',
        'source_size_bytes': 100,
        'replaces_submission_id': 'sub-old' if replacement else None,
    }
    counts = {'source': 0, 'credentials': 0, 'accept': 0}
    def source(*args, **kwargs):
        counts['source'] += 1
    def encrypt(*args, **kwargs):
        counts['credentials'] += 1
        return {'openrouter': 'ciphertext', 'deepline': 'ciphertext'}
    def accept(*args):
        counts['accept'] += 1
        return {'status': 'ok'}
    svc = ArenaService.__new__(ArenaService)
    svc._clock = lambda: moment
    svc._store = SimpleNamespace(get_submission=lambda _: row, accept_submission_with_credentials=accept)
    svc._config = SimpleNamespace(credential_manager=SimpleNamespace(validate_and_encrypt=encrypt))
    svc._validate_uploaded_source = source
    svc._enforce_submission_request_limit = lambda _: None
    svc._request_round = lambda *a, **kw: ({'hotkey': row['miner_hotkey'], 'body': {
        **{key: row[key] for key in ('submission_id', 'source_ref', 'source_size_bytes')},
        'credentials': {'openrouter_api_key': 'sk-or-v1-' + 'a' * 32,
                        'openrouter_management_key': 'sk-or-v1-' + 'b' * 32,
                        'deepline_api_key': 'deepline-' + 'c' * 32},
    }}, ROUND)
    return svc, counts


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_finalize_cutoff_is_exclusive_to_the_microsecond(delta):
    svc, counts = _service(CUTOFF + timedelta(microseconds=delta))
    if delta < 0:
        assert svc.handle_submission_finalize('sub-new', {})['status'] == 'accepted'
        assert counts == {'source': 1, 'credentials': 1, 'accept': 1}
    else:
        with pytest.raises(ServiceError, match='submission_replacement_closed'):
            svc.handle_submission_finalize('sub-new', {})
        assert counts == {'source': 0, 'credentials': 0, 'accept': 0}


def test_late_validation_cannot_bypass_database_cutoff():
    svc, counts = _service(CUTOFF - timedelta(microseconds=1))
    svc._store.accept_submission_with_credentials = lambda *args: {'status': 'replacement_closed'}
    with pytest.raises(ServiceError, match='submission_replacement_closed'):
        svc.handle_submission_finalize('sub-new', {})
    assert counts['credentials'] == 1


def test_first_submission_still_uses_midnight_cutoff():
    svc, counts = _service(CUTOFF + timedelta(minutes=30), replacement=False)
    assert svc.handle_submission_finalize('sub-new', {})['status'] == 'accepted'
    assert counts['accept'] == 1


@pytest.mark.parametrize('delta,expected', [(-1, 0), (0, 1), (1, 1)])
def test_review_does_not_start_while_sources_can_be_replaced(delta, expected):
    svc, _ = _service(CUTOFF + timedelta(microseconds=delta))
    reviewed = []
    svc._config.code_reviewer = SimpleNamespace(review=lambda row: reviewed.append(row) or {'status': 'passed'})
    svc.active_rounds = lambda: [ROUND]
    svc._round = lambda _: ROUND
    svc._store.list_submissions = lambda _, status: ([{'submission_id': 'sub-current', 'code_review_status': 'pending'}] if status == 'accepted' else [])
    assert svc.review_pending_submissions()['reviewed'] == expected
    assert len(reviewed) == expected


def test_replacement_deadline_matches_both_public_endpoints_and_is_not_cached():
    from tests.lab_arena.test_lab_arena_public_output_policy import _round, _service
    row = _round('arena-2026-09-16', 'open')
    row['configuration_doc']['schedule']['submission_cutoff'] = '2026-09-16T00:00:00.123456Z'
    with TestClient(create_app(_service([row]))) as http:
        current = http.get('/arena/v1/current')
        details = http.get('/arena/v1/rounds/arena-2026-09-16')
    expected = '2026-09-15T23:00:00.123456Z'
    assert current.json()['open_round']['submission_replacement_cutoff'] == expected
    assert details.json()['submission_replacement_cutoff'] == expected
    assert current.headers['cache-control'] == details.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('code', ['submission_replacement_closed', 'submission_window_closed'])
def test_final_database_deadline_failure_is_a_clear_conflict(code):
    svc, _ = _service(CUTOFF - timedelta(microseconds=1))
    def expired(*args):
        raise ArenaStoreError('lab_arena_' + code)
    svc._store.accept_submission_with_credentials = expired
    with TestClient(create_app(svc)) as http:
        response = http.post('/arena/v1/submissions/sub-new/finalize', json={'body': {'submission_id': 'sub-new'}})
    assert response.status_code == 409
    assert response.json()['code'] == code
