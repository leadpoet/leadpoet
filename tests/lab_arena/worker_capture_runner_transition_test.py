"""Production Runner/socket/final snapshot/completion transition, fake upstream only."""
import base64
from copy import deepcopy
import json
import os
import threading
from unittest.mock import patch

from lab_arena import contracts, lab_arena_checkpoint as checkpoint, runtime
from tests.lab_arena.trajectory_runner_test import LoggingApi, run_model
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, valid_company, verify
from tests.lab_arena.worker_final_capture_drain_test import _request_socket

BODY = b'{"choices":[]}'

class Api(LoggingApi):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.condition = threading.Condition()
        self.release = threading.Event()
        self.uploaded = []

    def provider(self, _run_id, _lease_token, frame):
        with self.condition:
            self.frames.append(deepcopy(frame))
            self.condition.notify_all()
        assert self.release.wait(8)
        return {'status': 200, 'headers': {}, 'body_b64': base64.b64encode(BODY).decode(),
                'call': {'operation_id': 'openrouter.chat', 'provider': 'openrouter',
                         'action_sequence': frame['action_sequence'],
                         'call_identity': contracts.document_hash(['offline', frame['action_sequence']]),
                         'outcome': 'settled', 'status': 200, 'provider_status': 200,
                         'actual_microusd': 0}}

    def training_observations(self, _run_id, _lease_token, document):
        self.uploaded.append(deepcopy(document))
        return {'status': 'stored', 'payload_sha256': contracts.document_hash(document)}

class BurstRuntime:
    def __init__(self, api, output):
        self.api, self.output = api, output
        self.capture_id = None
        self.provider_bodies = []

    def run_icp(self, spec, **_):
        connections = []
        try:
            for number in range(8):
                connections.append(_request_socket(spec.socket_path))
                with self.api.condition:
                    assert self.api.condition.wait_for(lambda: len(self.api.frames) == number + 1, timeout=2)
            with patch.dict(os.environ, {checkpoint.WORKER_SOCKET_ENV: str(spec.socket_path)}):
                self.capture_id = checkpoint.record_web_fetch(
                    url='https://example.test/page', final_url='https://example.test/page', status=200,
                    content_type='text/html', body='bounded final transition evidence', _retry_ambiguous=True)
        finally:
            self.api.release.set()
        for connection in connections:
            try:
                response = connection.getresponse()
                assert response.status == 200
                self.provider_bodies.append(response.read())
            finally:
                connection.close()
        return runtime.fake_result(output_bytes=json.dumps(self.output).encode())

def test_full_runner_retains_capture_without_changing_output(tmp_path):
    output = {'companies': [valid_company(1)]}
    normal_path, saturated_path = tmp_path / 'normal', tmp_path / 'saturated'
    normal_path.mkdir()
    saturated_path.mkdir()
    normal = Api()
    run_model(normal_path, normal, BridgingRuntime(output=output, calls=0))
    saturated = Api()
    sandbox = BurstRuntime(saturated, output)
    worker = run_model(saturated_path, saturated, sandbox)
    assert worker.abandoned == 0 and len(saturated.completions) == 1
    completion = saturated.completions[0]
    assert verify(completion['hotkey'], completion['signature'], contracts.signed_request_message(completion))
    body = completion['body']
    assert body['output'] == normal.completions[0]['body']['output']
    assert body['result']['terminal_status'] == 'accepted'
    assert body['result']['resource_summary']['provider_call_count'] == 8
    assert len(saturated.frames) == len(sandbox.provider_bodies) == 8
    assert all(value == BODY for value in sandbox.provider_bodies)
    payload = saturated.uploaded[-1]
    receipts = payload['response_receipts']
    assert len(receipts) == 8 and all(row['write_state'] == 'socket_written' for row in receipts)
    assert body['result']['resource_summary']['worker_observations']['payload_sha256'] == contracts.document_hash(payload)
    assert sandbox.capture_id == 0
    assert payload['complete']
    assert sum(row['kind'] == 'web_fetch' for row in payload['agent_observations']) == 1
