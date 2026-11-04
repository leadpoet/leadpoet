"""Capture diagnostics retain only fixed reasons; calls and deadlines stay bounded."""
import collections
import hashlib
import json
import queue
import socket
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from lab_arena import automatic_capture, lab_arena_checkpoint as checkpoint


def recorder(cp=checkpoint):
    value = automatic_capture.Recorder.__new__(automatic_capture.Recorder)
    value.checkpoint = cp
    value.lock = threading.RLock()
    value.counts = collections.Counter()
    value.handoff_dir = None
    value.queue = queue.Queue()
    value.inflight = 0
    value.adapters = set()
    return value


def read(c, n):
    data = b''
    while len(data) < n:
        part = c.recv(n-len(data))
        if not part:
            raise EOFError
        data += part
    return data


@pytest.mark.parametrize('mode,reason', [
    ('success', None), ('delayed', 'timeout'), ('rejected', 'host_rejected'),
    ('malformed', 'invalid_response'), ('bad_id', 'invalid_response'),
])
def test_socket_ack_diagnostic(tmp_path, monkeypatch, mode, reason):
    short = tempfile.TemporaryDirectory(prefix='cpr-', dir='/tmp')
    path = short.name+'/socket'
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    stored = []
    def serve():
        c, _ = server.accept()
        with c:
            size = int.from_bytes(read(c, 4), 'big') & 0x7fffffff
            frame = json.loads(read(c, size))
            checkpoint.validate_training_frame(frame)
            if mode != 'rejected':
                stored.append(frame)
            if mode == 'delayed':
                time.sleep(1.1)
            payload = (b'no' if mode == 'malformed' else json.dumps({
                'recorded': mode != 'rejected', 'observation_id': 'bad' if mode == 'bad_id' else 0,
            }).encode())
            try:
                c.sendall(len(payload).to_bytes(4, 'big')+payload)
            except BrokenPipeError:
                pass
    thread = threading.Thread(target=serve)
    thread.start()
    monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, path)
    raw = b'nonsecret text'
    body = SimpleNamespace(identity=None, truncated=False, size=len(raw), complete=True,
        content_type='text/plain', buffer=raw, encoding='utf-8', url='https://example.com',
        status=200, retrieved_at='2026-10-21T00:00:00+00:00', digest=hashlib.sha256(raw))
    checkpoint._TRAINING_FAILURE.reason = "host_rejected"
    value = recorder()
    value._enqueue(body)
    value._emit(value.queue.get_nowait())
    thread.join(2)
    server.close()
    short.cleanup()
    assert not thread.is_alive()
    assert checkpoint.TRAINING_SOCKET_TIMEOUT_SECONDS == 1.0
    assert checkpoint._training_failure_reason() == reason
    if reason:
        assert value.counts == {'checkpoint_failure': 1, 'checkpoint_failure_web_fetch_'+reason: 1}
    else:
        assert value.counts == {'web_written': 1}
    assert len(stored) == (mode != 'rejected')


def test_socket_invalid_frame_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, str(tmp_path/'missing'))
    assert checkpoint.record_web_fetch(url='https://example.com', final_url='https://example.com',
        status=200, content_type='text/plain', body='text') is None
    assert checkpoint._training_failure_reason() == 'transport_error'
    assert checkpoint._record_training('bad', {}, None) is None
    assert checkpoint._training_failure_reason() == 'invalid_frame'
    monkeypatch.delenv(checkpoint.WORKER_SOCKET_ENV)
    assert checkpoint.record_web_fetch(url='https://example.com', final_url='https://example.com',
        status=200, content_type='text/plain', body='text') is None
    assert checkpoint._training_failure_reason() == 'socket_unavailable'


def test_reason_is_thread_local_and_legacy_checkpoint_is_unknown():
    checkpoint._TRAINING_FAILURE.reason = 'timeout'
    def other():
        assert checkpoint._training_failure_reason() is None
        checkpoint._TRAINING_FAILURE.reason = 'host_rejected'
        return checkpoint._training_failure_reason()
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(other).result() == 'host_rejected'
    assert checkpoint._training_failure_reason() == 'timeout'
    value = recorder(SimpleNamespace())
    value._checkpoint_result('response_received', None)
    assert value.counts == {'checkpoint_failure': 1, 'checkpoint_failure_response_received_unknown': 1}


def test_parallel_capture_calls_keep_their_own_reason(tmp_path, monkeypatch):
    monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, str(tmp_path/'missing'))
    barrier = threading.Barrier(2)
    def invoke(invalid):
        if invalid:
            checkpoint._record_training('bad', {}, None)
        else:
            checkpoint.record_web_fetch(url='https://example.com', final_url='https://example.com',
                status=200, content_type='text/plain', body='text')
        barrier.wait(timeout=2)
        return checkpoint._training_failure_reason()
    with ThreadPoolExecutor(2) as pool:
        bad = pool.submit(invoke, True)
        missing = pool.submit(invoke, False)
        assert bad.result() == 'invalid_frame'
        assert missing.result() == 'transport_error'


def test_provider_ack_keeps_failure_and_incomplete(monkeypatch):
    monkeypatch.delenv(checkpoint.WORKER_SOCKET_ENV, raising=False)
    value = recorder()
    value._enqueue(SimpleNamespace(identity='sha256:'+'a'*64, complete=True,
        status=200, content_encoding='', provider_wire=False,
        digest=hashlib.sha256(b'text'), size=4))
    value._emit(value.queue.get_nowait())
    assert value.counts == {'checkpoint_failure': 1,
        'checkpoint_failure_response_received_socket_unavailable': 1}
    assert value._summary()['complete'] is False


@pytest.mark.parametrize("reason", ["https://private.example/secret", {}, []])
def test_untrusted_reason_does_not_enter_summary(reason):
    value = recorder(SimpleNamespace(_training_failure_reason=lambda: reason))
    value._checkpoint_result('web_fetch', None)
    assert value.counts == {'checkpoint_failure': 1, 'checkpoint_failure_web_fetch_unknown': 1}


def test_recursive_failure_report_success_does_not_clear_outer_reason(monkeypatch):
    sent = []
    payload = json.dumps({'recorded': True, 'observation_id': 7}).encode()
    class Connection:
        def __init__(self):
            self.response = len(payload).to_bytes(4, 'big')+payload
        def settimeout(self, value):
            pass
        def connect(self, path):
            pass
        def sendall(self, data):
            sent.append(json.loads(data[4:]))
        def recv(self, size):
            value, self.response = self.response[:size], self.response[size:]
            return value
        def close(self):
            pass
    monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, '/nonsecret/socket')
    monkeypatch.setattr(checkpoint.socket, 'socket', lambda *a: Connection())
    assert checkpoint._record_training('web_fetch', {}, None) is None
    assert checkpoint._training_failure_reason() == 'invalid_frame'
    assert len(sent) == 1
    assert sent[0]['kind'] == 'capture_failure'
