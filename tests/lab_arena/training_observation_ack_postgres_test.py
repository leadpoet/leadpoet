"""Real claimed baseline/miner leases retain capture after a delayed HTTP reply."""
import httpx
import pytest
from types import SimpleNamespace

from lab_arena import runner, training_observations
from lab_arena.service import LocalObjectStore
from tests.lab_arena import final_provider_judge_diagnostic_postgres_test as native
from tests.lab_arena.training_observation_ack_window_test import Clock, TimedCapture
from tests.lab_arena.training_observations_test import observation

scenario = native.scenario
reset = native.reset


@pytest.mark.parametrize('actor', ['baseline', 'miner'])
@pytest.mark.parametrize('legacy_timeout', [False, True], ids=['bounded-ack', 'old-lost-ack'])
def test_delayed_capture_preserves_native_lease_and_existing_results(scenario, tmp_path, monkeypatch, actor, legacy_timeout):
    context, rid, pending = native.base.prepare(scenario, attempt=5)
    if actor == 'miner':
        pending = native.miner_pending(context, pending)
    claimed = native.base.old.claim(context, pending=pending)
    assert claimed['status'] == 'leased' and claimed['run_id'] == pending['run_id']
    service = context[1].service
    # The direct SQL fixture claim omits the HTTP handler's capture binding.
    # Use the actual immutable database fields, not a synthetic claim hash.
    owned = service.store.get_run(claimed['run_id'])
    claimed.update(kind=owned['kind'], claim_request_hash=owned['claim_request_hash'])
    submission = service.store.get_submission(claimed['submission_id'])
    assert bool(submission.get('is_king')) == (actor == 'baseline')
    objects = LocalObjectStore(tmp_path / 'private-training')
    monkeypatch.setattr(service.config, 'training_object_store', objects)
    before = native.base.old.prior.snapshots(context)
    clock = Clock()
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=clock.monotonic))
    transport = TimedCapture(service, clock)
    token = claimed['lease_token']
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        api = runner.HttpArenaApiClient('http://localhost', client=client)
        if legacy_timeout:
            def legacy(run_id, lease_token, document, **_kwargs):
                return api._post('/arena/v1/runs/' + run_id + '/training-observation-segments', document,
                    headers={'x-lab-arena-lease': lease_token}, timeout_seconds=10.0,
                    preserve_http_status=True)
            monkeypatch.setattr(api, 'training_observation_segment', legacy)
        state = runner.RunState(lease=claimed, lease_token=token)
        state.worker_observations.append(observation())
        journal = runner._ObservationJournal(api, claimed, token, state)
        assert journal._prepare()
        acknowledged = journal._send_pending()
        assert acknowledged is (not legacy_timeout)
        assert clock.now == (10.0 if legacy_timeout else 45.0)
        if legacy_timeout:
            clock.advance(35.0)  # The real service commits after the client gave up.
    assert transport.committed_at == [45.0] and len(transport.requests) == 1
    assert journal.number == (0 if legacy_timeout else 1)
    assert (journal.pending is not None) == legacy_timeout
    segments, partial, error = training_observations.read_journal_prefix(objects,
        claimed['run_id'], 'score', claimed['lease_generation'], claimed['claim_request_hash'])
    assert error is None and len(segments) == len(partial['observations']) == 1
    assert partial['complete'] is False
    assert native.base.old.prior.snapshots(context) == before
