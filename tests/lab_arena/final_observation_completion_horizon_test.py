"""Final evidence remains replayable through the existing completion recovery."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import capture_retention, contracts, runner, training_observations as obs
from lab_arena.service import ServiceError
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config, valid_company
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for


def test_real_runner_replays_during_full_bounded_completion_recovery(tmp_path, monkeypatch):
    clock = [datetime(2026, 9, 2, 1, tzinfo=timezone.utc)]
    initial = clock[0]
    active = lease()
    active.update(lease_token=TOKEN, lease_expires_at=(initial + timedelta(seconds=120)).isoformat())
    service, objects = service_for(tmp_path)
    row = service._store.get_run('r1')
    row.update(round_id=active['round_id'], lease_expires_at=(initial + timedelta(seconds=3600)).isoformat(),
        claim_request_id='1' * 32, claim_response={'status': 'leased', 'request_id': '1' * 32,
            'run_id': 'r1', 'kind': 'execute', 'lease_generation': 1}, runner_hotkey='validator')
    service._clock = lambda: clock[0]
    service._store.get_round_capture_state = lambda _: {'round_id': row['round_id'], 'status': 'stage1'}

    def advance(seconds):
        clock[0] += timedelta(seconds=seconds)

    timing = SimpleNamespace(monotonic=lambda: (clock[0] - initial).total_seconds(), sleep=advance)
    monkeypatch.setattr(runner, 'time', timing)
    monkeypatch.setattr(capture_retention, 'time', timing)
    captures = []

    class Api(FakeApi):
        def __init__(self):
            super().__init__([active])
            self.final_calls = self.completion_calls = 0

        def training_observations(self, run, token, document, *, timeout_seconds=None):
            self.final_calls += 1
            if self.final_calls <= 2:
                advance(runner.TRAINING_OBSERVATIONS_TIMEOUT_SECONDS)
                raise httpx.ReadTimeout('synthetic foreground timeout')
            response = service.handle_training_observations(run, token, document)
            captures.append((timing.monotonic(), copy.deepcopy(document)))
            return response

        def verify_training_observations(self, *_):
            advance(runner.TRAINING_OBSERVATIONS_TIMEOUT_SECONDS)
            return False

        def complete(self, envelope):
            self.completion_calls += 1
            advance(runner.API_TIMEOUT_SECONDS)
            if self.completion_calls == 10:
                # The actual completion loop is still within its normal bound;
                # the current server lease remains authoritative during replay.
                config.input_retention._replay_batch()
            if self.completion_calls <= len(runner.COMPLETION_RECOVERY_RETRY_SECONDS):
                raise runner.RunnerError('synthetic completion availability failure', http_status=503)
            body = envelope['body']
            row.update(status='accepted', terminal_cause='accepted',
                result_doc=copy.deepcopy(body['result']), participation_accepted_at=clock[0].isoformat())
            return super().complete(envelope)

    api = Api()
    config = make_config(tmp_path, api, BridgingRuntime(output={'companies': [valid_company(1)]}, calls=0))
    config.clock = lambda: clock[0]
    config.completion_retry_seconds = runner.COMPLETION_RECOVERY_RETRY_SECONDS
    config.work_dir.mkdir()
    reads = [0]
    read = config.input_retention.observations._read

    def foreground_validation(*args):
        result = read(*args)
        reads[0] += 1
        if reads[0] <= 2:
            advance(8.3)
        return result

    monkeypatch.setattr(config.input_retention.observations, '_read', foreground_validation)
    assert runner.Runner(config).run_once() == 1
    assert api.completion_calls == 16 and len(api.completions) == 1
    assert api.final_calls == 3 and len(captures) == 1 and captures[0][0] > 360
    assert timing.monotonic() == pytest.approx(196.6 + runner.COMPLETION_RECOVERY_SECONDS)
    assert obs.read_snapshot(objects, 'r1', 'execute', 1, CLAIM_HASH)['payload'] == captures[0][1]
    assert not list((config.work_dir / 'pending-observations').glob('*.json'))
    assert not (config.work_dir / 'pending-observations' / 'audit').exists()
    assert api.provider_frames == []
    # Acceptance does not restart the terminal grace: the earlier signed
    # finished_at still prevents storage long after execution finished.
    before = copy.deepcopy(row)
    with pytest.raises(ServiceError) as denied:
        service.handle_training_observations('r1', TOKEN, captures[0][1])
    assert denied.value.status == 409 and row == before


def test_local_horizon_covers_existing_completion_budget_and_measured_overhead():
    existing_lifecycle = (3 * runner.TRAINING_OBSERVATIONS_TIMEOUT_SECONDS + 2 * 8.3
        + runner.COMPLETION_RECOVERY_SECONDS + 30)
    assert runner.COMPLETION_RECOVERY_SECONDS == 1152
    assert existing_lifecycle < contracts.LEASE_TTL_SECONDS
    assert capture_retention.FINAL_OBSERVATION_REPLAY_SECONDS == (
        contracts.LEASE_TTL_SECONDS + capture_retention.INPUT_GRACE_SECONDS)
