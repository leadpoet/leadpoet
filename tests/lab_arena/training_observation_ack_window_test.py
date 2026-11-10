"""Elapsed-time transport controls for accepted but delayed capture replies."""
from copy import deepcopy
import heapq
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import runner, training_observations as observations
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import observation


class Clock:
    def __init__(self):
        self.now = 0.0
        self.events = []
        self.sequence = 0

    def monotonic(self):
        return self.now

    def later(self, delay, callback):
        self.sequence += 1
        heapq.heappush(self.events, (self.now + delay, self.sequence, callback))

    def advance(self, seconds):
        target = self.now + seconds
        while self.events and self.events[0][0] <= target:
            self.now, _, callback = heapq.heappop(self.events)
            callback()
        self.now = target


class TimedCapture:
    """Advance elapsed time and finish server work even after client timeout."""
    def __init__(self, service, clock, *, lose_first_ack=False):
        self.service, self.clock = service, clock
        self.lose_first_ack = lose_first_ack
        self.requests, self.committed_at = [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.block_first = False

    def __call__(self, request):
        document = json.loads(request.content)
        run_id = request.url.path.split('/')[-2]
        token = request.headers['x-lab-arena-lease']
        if request.url.path.endswith('/training-observations'):
            return httpx.Response(200, json=self.service.handle_training_observations(run_id, token, document))
        assert request.url.path.endswith('/training-observation-segments')
        start, timeout = self.clock.now, request.extensions['timeout']['read']
        self.requests.append((deepcopy(document), start, timeout))
        result = {}
        def commit():
            result.update(self.service.handle_training_observation_segment(run_id, token, document))
            self.committed_at.append(self.clock.now)
        self.clock.later(45.0, commit)
        if self.block_first and len(self.requests) == 1:
            self.entered.set()
            assert self.release.wait(2), 'test did not release delayed reply'
        lost = self.lose_first_ack and len(self.requests) == 1
        # Timeouts are elapsed waits, not an immediate exception after a write.
        reply_at = start + 45.0
        end = start + timeout if lost or timeout < 45.0 else reply_at
        self.clock.advance(max(0.0, end - self.clock.now))
        if lost or timeout < 45.0:
            raise httpx.ReadTimeout('synthetic capture acknowledgment timeout', request=request)
        return httpx.Response(200, json=result)


def journal_for(tmp_path, monkeypatch, *, actor='baseline', kind='score', lose_first_ack=False):
    service, objects = service_for(tmp_path, kind=kind, submission_id=actor + '-submission',
                                   miner_hotkey=actor + '-hotkey')
    clock = Clock()
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=clock.monotonic))
    transport = TimedCapture(service, clock, lose_first_ack=lose_first_ack)
    client = httpx.Client(transport=httpx.MockTransport(transport))
    api = runner.HttpArenaApiClient('http://localhost', client=client)
    lease = dict(run_id='r1', kind=kind, lease_generation=1, claim_request_hash=CLAIM_HASH)
    state = runner.RunState(lease=lease, lease_token=TOKEN)
    state.worker_observations.append(observation())
    return runner._ObservationJournal(api, lease, TOKEN, state), transport, clock, objects, client


@pytest.mark.parametrize('actor', ['baseline', 'miner'])
@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_periodic_capture_receives_45_second_ack_without_reupload(tmp_path, monkeypatch, actor, kind):
    journal, transport, clock, objects, client = journal_for(tmp_path, monkeypatch, actor=actor, kind=kind)
    with client:
        assert journal._prepare()
        assert journal._send_pending()
    assert clock.now == 45.0 and transport.committed_at == [45.0]
    assert len(transport.requests) == journal.number == 1 and journal.pending is None
    segments, partial, error = observations.read_journal_prefix(objects, 'r1', kind, 1, CLAIM_HASH)
    assert error is None and len(segments) == partial['observation_count'] == 1
    assert partial['complete'] is False  # A journal is never terminal authority.


def test_genuinely_lost_ack_retries_same_bytes_after_commit(tmp_path, monkeypatch):
    journal, transport, clock, objects, client = journal_for(tmp_path, monkeypatch, lose_first_ack=True)
    with client:
        assert journal._prepare() and not journal._send_pending()
        assert clock.now == 60.0 and transport.committed_at == [45.0]
        assert journal.pending is not None and journal.number == 0
        clock.advance(journal.FLUSH_SECONDS)
        assert journal._send_pending()
    assert clock.now == 115.0 and transport.committed_at == [45.0, 115.0]
    assert len(transport.requests) == 2 and transport.requests[0][0] == transport.requests[1][0]
    assert journal.number == 1 and journal.pending is None
    assert len(observations.read_journal_prefix(objects, 'r1', 'score', 1, CLAIM_HASH)[0]) == 1


def test_terminal_send_uses_remaining_stop_allowance_and_keeps_final_authority(tmp_path, monkeypatch):
    journal, transport, clock, objects, client = journal_for(tmp_path, monkeypatch)
    for index in range(1, 65):
        journal.state.worker_observations.append(observation(ordinal=index, action=index))
    assert journal._prepare()
    journal.stop_event.set()
    journal.stop_deadline = 30.0
    with client:
        journal._flush_stopping()
        assert clock.now == 30.0 and len(transport.requests) == 1
        assert transport.requests[0][2] == 30.0 and journal.pending is not None
        summary = runner._record_training_observations(SimpleNamespace(api=journal.api),
            journal.lease, TOKEN, journal.state)
        assert summary['status'] == 'stored'
        clock.advance(15.0)  # The timed-out server write commits after snapshot freeze.
        journal._flush_stopping()
    assert transport.committed_at == [45.0] and len(transport.requests) == 1
    assert journal.number == 0  # No acknowledgment was received.
    prefix = observations.read_journal_prefix(objects, 'r1', 'score', 1, CLAIM_HASH)[1]
    final = observations.read_snapshot(objects, 'r1', 'score', 1, CLAIM_HASH)['payload']
    assert len(prefix['observations']) == 64 and prefix['complete'] is False
    assert final['observation_count'] == 65 and final['complete'] is True


def test_periodic_late_ack_cannot_send_new_delta_after_stop_and_freeze(tmp_path, monkeypatch):
    journal, transport, clock, objects, client = journal_for(tmp_path, monkeypatch)
    transport.block_first = True
    assert journal._prepare()
    def upload():
        try:
            journal._send_pending()
        finally:
            journal._flush_stopping()
    upload_thread = threading.Thread(target=upload)
    upload_thread.start()
    assert transport.entered.wait(2)
    class JoinAtDeadline:
        ident = upload_thread.ident
        def join(self, timeout):
            clock.advance(timeout)
    journal.thread = JoinAtDeadline()
    try:
        journal.state.worker_observations.append(observation(ordinal=1, action=1))
        journal.stop()
        assert clock.now == 30.0 and upload_thread.is_alive()
        summary = runner._record_training_observations(SimpleNamespace(api=journal.api),
            journal.lease, TOKEN, journal.state)
        assert summary['status'] == 'stored' and journal.state.worker_observations_frozen
        transport.release.set()
        upload_thread.join(2)
        assert not upload_thread.is_alive()
        assert clock.now == 45.0 and journal.number == 1 and len(transport.requests) == 1
        assert observations.read_snapshot(objects, 'r1', 'score', 1, CLAIM_HASH)['payload']['observation_count'] == 2
        assert observations.read_journal_prefix(objects, 'r1', 'score', 1, CLAIM_HASH)[1]['observation_count'] == 1
    finally:
        transport.release.set()
        upload_thread.join(2)
        client.close()


def test_dispatch_rechecks_stop_budget_before_starting_http_request(tmp_path, monkeypatch):
    journal, transport, clock, objects, client = journal_for(tmp_path, monkeypatch)
    assert journal._prepare()
    journal.stop_event.set()
    journal.stop_deadline = 9.0  # The existing ten-second minimum admission is retained.
    with client:
        assert not journal._send_pending()
    assert clock.now == 0.0 and transport.requests == [] and journal.number == 0
