"""Thread startup rollback must preserve relay and context cleanup (local sockets only)."""

import os
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from lab_arena import agent_entrypoint
from lab_arena import web_egress_bridge as module


@pytest.fixture
def echo_socket():
    with tempfile.TemporaryDirectory(prefix="bridge-start-", dir="/tmp") as directory:
        path = Path(directory) / "echo.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(2)
        listener.settimeout(0.1)
        stopped = threading.Event()

        def serve():
            while not stopped.is_set():
                try:
                    client, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                with client:
                    client.settimeout(2)
                    data = client.recv(1024)
                    if data:
                        client.sendall(data[::-1])

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield path
        finally:
            stopped.set()
            listener.close()
            thread.join(timeout=3)
            assert not thread.is_alive()


def assert_clean(bridge):
    assert bridge._workers == set()
    assert bridge._connections == set()
    assert bridge._listener is None
    assert bridge._thread is None
    assert bridge._slots.acquire(blocking=False)
    assert not bridge._slots.acquire(blocking=False)
    bridge._slots.release()


@pytest.mark.parametrize("failure_at", ["construct", "start"])
def test_failed_worker_start_releases_resources_and_accepts_next_client(
    echo_socket, monkeypatch, failure_at
):
    bridge = module.LoopbackWebEgressBridge(echo_socket, max_connections=1).start()
    original_thread = threading.Thread
    original_start = threading.Thread.start
    failures = []

    def fail_once(name):
        if name == "arena-web-egress-bridge-client" and not failures:
            failures.append(name)
            raise RuntimeError("can't start new thread")

    def construct(*args, **kwargs):
        fail_once(kwargs.get("name"))
        return original_thread(*args, **kwargs)

    def start(thread):
        fail_once(thread.name)
        return original_start(thread)

    if failure_at == "construct":
        monkeypatch.setattr(module.threading, "Thread", construct)
    else:
        monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with socket.create_connection(("127.0.0.1", bridge.port), timeout=2) as failed:
            assert failed.recv(1) == b""
        assert failures == ["arena-web-egress-bridge-client"]
        assert bridge.last_failure == {
            "stage": "worker_start", "error_type": "RuntimeError", "errno": 0
        }
        assert bridge.running
        with socket.create_connection(("127.0.0.1", bridge.port), timeout=2) as client:
            client.sendall(b"relay after failure")
            client.shutdown(socket.SHUT_WR)
            assert client.recv(1024) == b"eruliaf retfa yaler"
            assert client.recv(1) == b""
    finally:
        bridge.stop()
    assert_clean(bridge)


@pytest.mark.parametrize("failure_at", ["construct", "start"])
def test_listener_start_failure_rolls_back_and_preserves_original_error(
    echo_socket, monkeypatch, failure_at
):
    bridge = module.LoopbackWebEgressBridge(echo_socket, max_connections=1)
    original_thread = threading.Thread
    original_start = threading.Thread.start
    failure = RuntimeError("can't start new thread")

    def construct(*args, **kwargs):
        if kwargs.get("name") == "arena-web-egress-bridge":
            raise failure
        return original_thread(*args, **kwargs)

    def start(thread):
        if thread.name == "arena-web-egress-bridge":
            raise failure
        return original_start(thread)

    with monkeypatch.context() as patch:
        if failure_at == "construct":
            patch.setattr(module.threading, "Thread", construct)
        else:
            patch.setattr(threading.Thread, "start", start)
        with pytest.raises(RuntimeError) as caught:
            bridge.start()
        assert caught.value is failure
        assert bridge.last_failure == {
            "stage": "listener_start", "error_type": "RuntimeError", "errno": 0
        }
        failed_port = bridge.port
        bridge.stop()
        assert_clean(bridge)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", failed_port), timeout=0.2)
    # Rollback leaves the same bridge usable once thread creation recovers.
    bridge.start()
    bridge.stop()
    assert_clean(bridge)


def test_unexpected_worker_start_error_rolls_back_and_propagates(monkeypatch):
    bridge = module.LoopbackWebEgressBridge("/tmp/unused-bridge.sock", max_connections=1)
    client, peer = socket.socketpair()
    failure = ValueError("unexpected thread setup error")

    class Listener:
        def accept(self):
            return client, ("127.0.0.1", 1)

    def construct(*args, **kwargs):
        raise failure

    bridge._listener = Listener()
    monkeypatch.setattr(module.threading, "Thread", construct)
    try:
        with pytest.raises(ValueError) as caught:
            bridge._serve()
        assert caught.value is failure
        assert client.fileno() == -1
        assert peer.recv(1) == b""
        bridge.stop()
        assert_clean(bridge)
    finally:
        peer.close()


@pytest.mark.parametrize("fail_start", [False, True])
def test_stop_cannot_join_worker_until_start_or_rollback_finishes(
    echo_socket, monkeypatch, fail_start
):
    bridge = module.LoopbackWebEgressBridge(echo_socket, max_connections=1).start()
    original_start = threading.Thread.start
    entered = threading.Event()
    release = threading.Event()
    stop_called = threading.Event()
    errors = []

    def start(thread):
        if thread.name == "arena-web-egress-bridge-client":
            entered.set()
            assert release.wait(timeout=3)
            if fail_start:
                raise RuntimeError("can't start new thread")
        return original_start(thread)

    def stop():
        stop_called.set()
        try:
            bridge.stop()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(threading.Thread, "start", start)
    client = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
    stopper = threading.Thread(target=stop)
    try:
        assert entered.wait(timeout=2)
        stopper.start()
        assert stop_called.wait(timeout=2)
    finally:
        release.set()
        stopper.join(timeout=5)
        client.close()
    assert not stopper.is_alive()
    assert errors == []
    assert_clean(bridge)


def test_stop_rejects_accepted_client_before_worker_start(echo_socket, monkeypatch):
    bridge = module.LoopbackWebEgressBridge(echo_socket, max_connections=1).start()
    original_acquire = bridge._slots.acquire
    original_start = threading.Thread.start
    acquired = threading.Event()
    release = threading.Event()
    worker_starts = []
    errors = []

    def acquire(*args, **kwargs):
        result = original_acquire(*args, **kwargs)
        acquired.set()
        assert release.wait(timeout=3)
        return result

    def start(thread):
        if thread.name == "arena-web-egress-bridge-client":
            worker_starts.append(thread)
        return original_start(thread)

    def stop():
        try:
            bridge.stop()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(bridge._slots, "acquire", acquire)
    monkeypatch.setattr(threading.Thread, "start", start)
    client = socket.create_connection(("127.0.0.1", bridge.port), timeout=2)
    stopper = threading.Thread(target=stop)
    try:
        assert acquired.wait(timeout=2)
        stopper.start()
        assert bridge._stop.wait(timeout=2)
    finally:
        release.set()
        stopper.join(timeout=5)
        client.close()
    assert not stopper.is_alive()
    assert errors == []
    assert worker_starts == []
    assert_clean(bridge)


def test_entrypoint_keeps_completed_output_after_worker_start_failure(
    echo_socket, monkeypatch, tmp_path
):
    original_start = threading.Thread.start
    failure_seen = threading.Event()

    def start(thread):
        if thread.name == "arena-web-egress-bridge-client":
            failure_seen.set()
            raise RuntimeError("can't start new thread")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(agent_entrypoint, "WEB_BRIDGE_PATH", Path(module.__file__))
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", str(echo_socket))
    monkeypatch.setenv("HTTP_PROXY", "http://previous.invalid")
    output = tmp_path / "output.json"
    with agent_entrypoint._web_access():
        port = int(os.environ["HTTP_PROXY"].rsplit(":", 1)[1])
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            assert failure_seen.wait(timeout=2)
            output.write_text('{"completed": true}')
    assert output.read_text() == '{"completed": true}'
    assert os.environ["HTTP_PROXY"] == "http://previous.invalid"


def test_real_incomplete_worker_cleanup_still_raises(echo_socket, monkeypatch):
    bridge = module.LoopbackWebEgressBridge(
        echo_socket, max_connections=1, connect_timeout_seconds=0.1
    )
    original_run = bridge._run_client
    entered = threading.Event()
    release = threading.Event()

    def run(client):
        entered.set()
        release.wait(timeout=5)
        original_run(client)

    monkeypatch.setattr(bridge, "_run_client", run)
    bridge.start()
    try:
        with socket.create_connection(("127.0.0.1", bridge.port), timeout=2):
            assert entered.wait(timeout=2)
            with pytest.raises(
                module.LoopbackWebEgressBridgeError, match="cleanup is incomplete"
            ):
                bridge.stop()
    finally:
        release.set()
        bridge.stop()
    assert_clean(bridge)
