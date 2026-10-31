"""The trusted shim must leave time to return an in-budget Arena API reply."""
import base64
import http.server
import json
from pathlib import Path
import tempfile
import threading
import time

import pytest

from lab_arena import contracts, operations, runner, shim
from tests.lab_arena.test_lab_arena_runner import lease


BODY = b'{"choices":[]}'
IDENTITY = contracts.hash_bytes(b'local shim return-window test')


@pytest.mark.parametrize('worker_delay,provider_delay', [(0.01, 0.02), (0.5, 1.1)])
def test_real_shim_keeps_in_budget_api_response_until_host_write(
    monkeypatch, worker_delay, provider_delay,
):
    # Scale only transport budgets and controlled latency. Keep real sockets,
    # the HTTP API client, frame handler, response capture and socket writes.
    monkeypatch.setattr(operations, 'BUDGET_ADMISSION_MAX_SECONDS', 0.4)
    monkeypatch.setattr(operations, 'PROVIDER_BILLING_RECONCILIATION_SECONDS', 0.6)
    monkeypatch.setattr(operations, 'PROVIDER_API_TIMEOUT_GRACE_SECONDS', 0.3)
    monkeypatch.setattr(runner, 'PROVIDER_API_TIMEOUT_GRACE_SECONDS', 0.3)
    monkeypatch.setattr(runner, 'API_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setattr(shim, 'SOCKET_GRACE_SECONDS', 0.3)
    api_frames = []
    upstream_seconds = []

    class Provider(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            frame = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            api_frames.append(frame)
            started = time.monotonic()
            time.sleep(provider_delay)
            document = {'status': 200, 'headers': {},
                'body_b64': base64.b64encode(BODY).decode(),
                'call': {'operation_id': 'openrouter.chat', 'provider': 'openrouter',
                         'action_sequence': frame['action_sequence'],
                         'call_identity': IDENTITY, 'outcome': 'settled',
                         'status': 200, 'provider_status': 200}}
            encoded = contracts.canonical_json(document).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            upstream_seconds.append(time.monotonic() - started)

    provider = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    api = runner.HttpArenaApiClient('http://127.0.0.1:%d' % provider.server_port)
    api_timeouts = []
    api._client.event_hooks['request'].append(
        lambda request: api_timeouts.append(dict(request.extensions['timeout'])))
    original_provider = api.provider

    def delayed_provider(*args, **kwargs):
        # Local dispatch/scheduling before the upstream request starts. Its
        # duration is outside httpx's per-request timeout, but inside the
        # already-waiting sandbox socket's read timeout.
        time.sleep(worker_delay)
        return original_provider(*args, **kwargs)

    monkeypatch.setattr(api, 'provider', delayed_provider)
    state = runner.RunState(lease=lease(), lease_token='local-test-token')
    with tempfile.TemporaryDirectory(prefix='ar-shim-window-', dir='/tmp') as directory:
        server = runner.WorkerSocketServer(Path(directory) / 'worker.sock', api, state)
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(server._path))
        server.start()
        try:
            try:
                result = shim.dispatch('openrouter.chat', {
                    'model': 'openai/gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': 'local return-window test'}],
                    'max_tokens': 8}, 200)
                transport_error = None
            except shim.ShimTransportError as exc:
                result = None
                transport_error = str(exc)
            assert server.wait_for_handlers()
            # No paid-call retry or changed frame/provider deadline is needed.
            assert len(api_frames) == len(state.calls) == state.action_sequence == 1
            assert api_frames[0]['timeout_ms'] == 200
            assert len(api_timeouts) == 1
            assert api_timeouts[0]['read'] == pytest.approx(1.5)
            assert upstream_seconds and upstream_seconds[0] < api_timeouts[0]['read']
            payload = state.worker_observations.document()
            receipt = payload['response_receipts'][0]
            assert transport_error is None, (
                'trusted shim closed before in-budget upstream reply; '
                'actual host write state: ' + receipt['write_state'])
            assert result[0] == 200 and result[2] == BODY
            assert result[1][shim.CALL_IDENTITY_HEADER] == IDENTITY
            assert receipt['write_state'] == 'socket_written'
            assert receipt['body_sha256'] == contracts.hash_bytes(BODY)
            assert payload['complete'] is True and payload['failures'] == []
        finally:
            server.stop()
            server.wait_for_handlers()
            api._client.close()
            provider.shutdown()
            provider.server_close()
            provider_thread.join(2)
