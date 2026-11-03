"""Real loopback HTTP admission, with no provider or model network calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import socket
import struct
import threading
from urllib.parse import urlsplit

import httpx
import pytest

from lab_arena import lab_arena_codex as codex
from tests.lab_arena.codex_runtime_test import response


REQUEST = {"model": "openai/gpt-4o-mini", "input": "local fixture", "stream": False}
VALID_RESPONSE = json.dumps(response()).encode()


def post(bridge, document, path="/responses", **kwargs):
    with httpx.Client(trust_env=False, timeout=3) as client:
        return client.post(bridge.base_url + path,
            headers={"Authorization": "Bearer " + bridge.token}, json=document, **kwargs)


@contextmanager
def paused_first_reply(bridge, monkeypatch):
    """Hold handler cleanup after the complete reply reaches its client.

    This deterministically exposes the old reply-before-unlock window without
    a sleep or an assumption about thread scheduling between two requests.
    """
    handler = bridge.server.RequestHandlerClass
    original = handler.reply
    written, finish = threading.Event(), threading.Event()
    count_lock = threading.Lock()
    count = 0

    def reply(self, *args, **kwargs):
        nonlocal count
        with count_lock:
            count += 1
            first = count == 1
        original(self, *args, **kwargs)
        if first:
            written.set()
            assert finish.wait(5), "test did not release completed handler"

    monkeypatch.setattr(handler, "reply", reply)
    try:
        yield written
    finally:
        finish.set()


@pytest.mark.parametrize("first_kind,expected_status", [
    ("invalid", 400), ("success", 200), ("sse", 200), ("invalid_sse", 502),
    ("provider_error", 404), ("dispatch_error", 502), ("transport_error", 502),
    ("guard_denied", 429), ("guard_error", 429),
    ("standalone_miss", 200), ("standalone_search", 200), ("standalone_page", 200),
])
@pytest.mark.parametrize("next_valid", [False, True])
def test_complete_reply_admits_next_sequential_request_before_handler_cleanup(
    monkeypatch, first_kind, expected_status, next_valid,
):
    dispatches = []
    def dispatch(*args, **kwargs):
        dispatches.append(True)
        if first_kind == "dispatch_error" and len(dispatches) == 1:
            raise codex.CodexRuntimeError("fixture failure")
        if first_kind == "transport_error" and len(dispatches) == 1:
            raise OSError("fixture disconnect")
        if first_kind == "invalid_sse" and len(dispatches) == 1:
            return 200, b'{"invalid":"response"}'
        if first_kind == "provider_error" and len(dispatches) == 1:
            return 404, b'{"error":{"message":"fixture rejection"}}'
        return 200, VALID_RESPONSE

    def guard():
        if first_kind == "guard_error":
            raise RuntimeError("fixture guard failure")
        return first_kind != "guard_denied"

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    monkeypatch.setattr(codex, "_standalone_page_output", lambda *args, **kwargs: b'{"output":"local page"}')
    bridge = codex.ResponsesBridge('/unused-no-provider.sock', web_search='live', request_guard=guard)
    first = dict(REQUEST)
    path = '/responses'
    if first_kind == 'invalid':
        first['stream'] = 'invalid'
    elif first_kind in ('sse', 'invalid_sse'):
        first['stream'] = True
    elif first_kind.startswith('standalone_'):
        path = '/alpha/search'
        first = {'model': REQUEST['model'], 'commands': {}}
        if first_kind == 'standalone_search':
            first['commands'] = {'search_query': [{'q': 'local fixture'}]}
        elif first_kind == 'standalone_page':
            first['commands'] = {'open': [{'ref_id': 'https://example.com/'}]}
    with bridge, paused_first_reply(bridge, monkeypatch) as written:
        initial = post(bridge, first, path)
        assert initial.status_code == expected_status
        assert written.wait(1)
        # The first request is complete even though its server thread is still
        # paused. A new bad schema is400, not a fabricated busy429.
        next_request = dict(REQUEST, stream=False if next_valid else 'invalid')
        if first_kind in ('guard_denied', 'guard_error'):
            bridge._request_guard = None
        following = post(bridge, next_request)
        assert following.status_code == (200 if next_valid else 400)
        assert bridge.wait_idle(0) is True


def test_genuinely_overlapping_requests_and_early_errors_cannot_release_owner(monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    dispatches = []
    def dispatch(*args, **kwargs):
        dispatches.append(True)
        entered.set()
        assert finish.wait(5)
        return 200, VALID_RESPONSE
    monkeypatch.setattr(codex, '_dispatch', dispatch)
    bridge = codex.ResponsesBridge('/unused-no-provider.sock')
    try:
        with bridge, ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(post, bridge, REQUEST)
            assert entered.wait(2)
            with httpx.Client(trust_env=False, timeout=3) as client:
                assert client.post(bridge.base_url+'/unsupported', json={}).status_code == 404
                assert client.post(bridge.base_url+'/responses', json={}).status_code == 401
            for _ in range(2):
                assert post(bridge, REQUEST).status_code == 429
                assert bridge.wait_idle(0) is False
            assert len(dispatches) == 1
            finish.set()
            assert active.result(timeout=3).status_code == 200
            assert post(bridge, dict(REQUEST, stream='invalid')).status_code == 400
    finally:
        finish.set()


def test_previous_handler_cleanup_cannot_release_next_active_request(monkeypatch):
    first_written, first_cleanup, first_finished = (threading.Event() for _ in range(3))
    second_dispatch, finish_dispatch = threading.Event(), threading.Event()
    dispatches = []
    def dispatch(*args, **kwargs):
        dispatches.append(True)
        second_dispatch.set()
        assert finish_dispatch.wait(5)
        return 200, VALID_RESPONSE
    monkeypatch.setattr(codex, '_dispatch', dispatch)
    bridge = codex.ResponsesBridge('/unused-no-provider.sock')
    handler = bridge.server.RequestHandlerClass
    original_reply, original_post = handler.reply, handler.do_POST
    def reply(self, *args, **kwargs):
        original_reply(self, *args, **kwargs)
        if not first_written.is_set():
            self._first_fixture_handler = True
            first_written.set()
            assert first_cleanup.wait(5)
    def do_POST(self):
        try:
            original_post(self)
        finally:
            if getattr(self, '_first_fixture_handler', False):
                first_finished.set()
    monkeypatch.setattr(handler, 'reply', reply)
    monkeypatch.setattr(handler, 'do_POST', do_POST)
    with bridge, ThreadPoolExecutor(max_workers=1) as pool:
        try:
            assert post(bridge, dict(REQUEST, stream='invalid')).status_code == 400
            assert first_written.wait(1)
            second = pool.submit(post, bridge, REQUEST)
            assert second_dispatch.wait(2)
            first_cleanup.set()
            assert first_finished.wait(2)
            # The previous handler's finally has now run while this second
            # handler still owns admission. It must not open a third dispatch.
            assert bridge.wait_idle(0) is False
            assert post(bridge, REQUEST).status_code == 429
            assert len(dispatches) == 1
            finish_dispatch.set()
            assert second.result(timeout=3).status_code == 200
        finally:
            first_cleanup.set()
            finish_dispatch.set()


def test_client_disconnect_releases_owner_once_and_next_request_is_admitted(monkeypatch):
    entered, finish, handler_finished = threading.Event(), threading.Event(), threading.Event()
    def dispatch(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return 200, VALID_RESPONSE
    monkeypatch.setattr(codex, '_dispatch', dispatch)
    bridge = codex.ResponsesBridge('/unused-no-provider.sock')
    releases = []
    original_release = bridge._active.release
    def release():
        releases.append(True)
        original_release()
    monkeypatch.setattr(bridge._active, 'release', release)
    original = bridge.server.RequestHandlerClass.do_POST
    def do_POST(self):
        try:
            original(self)
        finally:
            handler_finished.set()
    monkeypatch.setattr(bridge.server.RequestHandlerClass, 'do_POST', do_POST)
    # A reset peer can reject both the original write and its502 fallback.
    monkeypatch.setattr(bridge.server, 'handle_error', lambda *_: None)
    try:
        with bridge:
            target = urlsplit(bridge.base_url)
            client = socket.create_connection((target.hostname, target.port), timeout=2)
            raw = json.dumps(REQUEST).encode()
            client.sendall(('POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n'
                'Authorization: Bearer '+bridge.token+'\r\nContent-Length: '+str(len(raw))+
                '\r\nConnection: close\r\n\r\n').encode()+raw)
            assert entered.wait(2)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            client.close()
            finish.set()
            assert handler_finished.wait(2)
            assert len(releases) == 1
            assert bridge.wait_idle(0) is True
            assert post(bridge, dict(REQUEST, stream='invalid')).status_code == 400
    finally:
        finish.set()


def test_unhandled_processing_failure_keeps_finally_release_fallback(monkeypatch):
    def dispatch(*args, **kwargs):
        raise RuntimeError('unexpected fixture processing failure')
    monkeypatch.setattr(codex, '_dispatch', dispatch)
    bridge = codex.ResponsesBridge('/unused-no-provider.sock')
    errors = []
    monkeypatch.setattr(bridge.server, 'handle_error', lambda *_: errors.append(True))
    with bridge:
        with pytest.raises(httpx.RemoteProtocolError):
            post(bridge, REQUEST)
        assert bridge.wait_idle(1) is True
        assert post(bridge, dict(REQUEST, stream='invalid')).status_code == 400
    assert errors
