"""Provider transport diagnostics retain safe cause classes, never request text."""

from datetime import datetime, timezone
import json
import socket
import threading

import httpx
import pytest

from lab_arena import runner, trajectory
from tests.lab_arena.test_lab_arena_runner import lease


def test_real_http_socket_interruption_retains_cause_and_original_event_time(tmp_path):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(3)
    host, port = listener.getsockname()
    received = []
    failures = []

    def interrupt():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(3)
                request = bytearray()
                while b'\r\n\r\n' not in request:
                    chunk = connection.recv(65536)
                    if not chunk:
                        raise AssertionError('request headers incomplete')
                    request.extend(chunk)
                headers, body = request.split(b'\r\n\r\n', 1)
                length = next(int(line.split(b':', 1)[1]) for line in headers.split(b'\r\n')
                              if line.lower().startswith(b'content-length:'))
                while len(body) < length:
                    chunk = connection.recv(65536)
                    if not chunk:
                        raise AssertionError('request body incomplete')
                    body.extend(chunk)
                received.append(bytes(body))
                # The request reached the HTTP peer. Close before any response;
                # this proves neither provider dispatch nor a zero charge.
                connection.shutdown(socket.SHUT_RDWR)
        except Exception as exc:
            failures.append(type(exc).__name__)

    thread = threading.Thread(target=interrupt, daemon=True)
    thread.start()
    api = runner.HttpArenaApiClient(f'http://{host}:{port}')
    state = runner.RunState(lease=lease(), lease_token='private-lease-secret')
    worker = runner.WorkerSocketServer(tmp_path / 'worker.sock', api, state)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    try:
        error, result = worker._dispatch_once('deepline.execute',
            {'tool': 'exa_search', 'payload': {'query': 'private-parameter-secret'}}, 1000)
    finally:
        api.close()
        listener.close()
        thread.join(timeout=4)
    assert not thread.is_alive() and not failures and len(received) == 1
    assert error == 'worker_unavailable' and result is None
    assert len(state.calls) == state.action_sequence == 1
    assert state.calls[0]['outcome'] == 'unknown' and 'call_identity' not in state.calls[0]
    event, = runner._provider_error_events(state)
    assert event['content']['transport_cause'] == 'RemoteProtocolError'
    assert event['content']['outcome'] == 'unknown' and 'http_status' not in event['content']
    occurred_at = datetime.fromisoformat(event['occurred_at'].replace('Z', '+00:00'))
    assert before <= occurred_at <= datetime.now(timezone.utc)
    # Buffered export and upload retries must retain the failure's original clock.
    assert runner._provider_error_events(state) == [event]
    trajectory.validate_batch({'events': [event]})
    saved = json.dumps(event)
    assert 'private-lease-secret' not in saved and 'private-parameter-secret' not in saved
    assert f'http://{host}:{port}' not in saved


@pytest.mark.parametrize('cause,expected', [
    (httpx.ReadTimeout('private exception prose https://private.example/path'), 'ReadTimeout'),
    (httpx.ConnectError('private exception prose'), 'ConnectError'),
    (type('PrivateCredentialLikeToken123', (httpx.HTTPError,), {})('private prose'), 'unknown'),
    (type('ReadTimeout', (httpx.HTTPError,), {})('private prose'), 'unknown'),
    (type('bad-class https://private.example', (httpx.HTTPError,), {})('private prose'), 'unknown'),
    (type('X' * 80, (httpx.HTTPError,), {})('private prose'), 'unknown'),
    (ValueError('private exception prose'), None),
])
def test_only_bounded_httpx_cause_class_is_retained(cause, expected):
    exc = runner.RunnerError('private outer exception prose')
    exc.__cause__ = cause
    content = runner._provider_error_event('deepline.execute', 7, exc)['content']
    assert content.get('transport_cause') == expected
    assert 'private' not in json.dumps(content)
    assert content['outcome'] == 'unknown'
