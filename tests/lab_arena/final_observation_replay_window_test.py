"""Foreground capture timeouts must not exhaust local replay before completion."""
import copy
from datetime import datetime, timedelta, timezone
import json

import httpx
import pytest

from lab_arena import capture_retention, runner, scoring, training_observations as obs
from tests.lab_arena.input_retention_test import finish_poll
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config, scoring_lease, valid_company,
)
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("restart", [False, True])
def test_runner_final_timeouts_leave_replay_with_open_terminal_grace(tmp_path, kind, restart):

    clock = [datetime(2026, 9, 2, 1, tzinfo=timezone.utc)]
    start = clock[0]
    active = scoring_lease(run_id='r1') if kind == "score" else lease()
    active['lease_token'] = TOKEN
    service, objects = service_for(tmp_path, kind=kind)
    row = service._store.get_run('r1')
    row.update(round_id=active['round_id'], lease_expires_at=active['lease_expires_at'],
        claim_request_id='1'*32, claim_response={'status':'leased','request_id':'1'*32,
            'run_id':'r1','kind':kind,'lease_generation':1}, runner_hotkey='validator')
    service._clock = lambda: clock[0]
    service._store.get_round_capture_state = lambda _: {'round_id': row['round_id'], 'status':'stage1'}

    class Api(FakeApi):
        def __init__(self):
            super().__init__([active])
            self.final_calls = 0

        def training_observations(self, run, token, document, *, timeout_seconds=None):
            self.final_calls += 1
            if self.final_calls <= 2:
                clock[0] += timedelta(seconds=60)
                raise httpx.ReadTimeout('synthetic lost capture request')
            return service.handle_training_observations(run, token, document)

        def verify_training_observations(self, *_):
            clock[0] += timedelta(seconds=60)
            return False

        def complete(self, envelope):
            assert super().complete(envelope)['status'] == 'accepted'
            body = envelope['body']
            row.update(status='accepted', terminal_cause='accepted',
                result_doc=copy.deepcopy(body['result']), participation_accepted_at=clock[0].isoformat())
            return {'status':'accepted'}

    api = Api()
    output = (scoring.build_scoring_output('r1', [{'final_score': 71.0, 'failure_reason': ''},
        {'final_score': 44.5, 'failure_reason': ''}]) if kind == 'score' else {'companies':[valid_company(1)]})
    sandbox = BridgingRuntime(output=output, calls=0)
    config = make_config(tmp_path, api, sandbox)
    config.clock = lambda: clock[0]
    config.work_dir.mkdir()
    assert runner.Runner(config).run_once() == 1
    assert clock[0] == start + timedelta(seconds=180)
    assert api.completions[0]['body']['result']['finished_at'] == clock[0].isoformat().replace('+00:00','Z')
    path = next((config.work_dir/'pending-observations').glob('*.json'))
    original = path.read_bytes()
    # Same token, same claim, unchanged terminal capture-only server authority remains valid.
    assert service._training_final_observation_capture_run('r1', TOKEN) == row
    before = copy.deepcopy(row)
    completion = copy.deepcopy(api.completions[0])
    if restart:
        config.input_retention = capture_retention.InputRetention(config)
    finish_poll(config.input_retention)
    assert row == before
    assert api.final_calls == 3
    assert obs.read_snapshot(objects, 'r1', kind, 1, CLAIM_HASH)['payload'] == json.loads(original)['observations']

    assert len(api.completions) == 1 and api.provider_frames == []
    assert api.completions[0] == completion
    if kind == 'score':
        assert completion['body']['output'] == output
    finish_poll(config.input_retention)
    assert api.final_calls == 3 and len(api.completions) == 1
