"""The chain relay's accept loop must survive transient accept() errors.

The relay is the enclave's only chain-RPC path (every weight-extrinsic
signature routes through it). Before this fix, one transient accept() error
returned from the loop and main() never relaunched it — signing died
silently every epoch until a manual restart. These tests pin: a transient
error keeps the loop serving, a genuine connection is still handled after
one, and stop() still exits cleanly.
"""
import importlib.util
import pathlib
import socket
import threading
import time

_spec = importlib.util.spec_from_file_location(
    "chain_relay_v2",
    pathlib.Path(__file__).resolve().parents[1]
    / "validator_tee" / "host" / "chain_relay_v2.py",
)
relay_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay_mod)
ValidatorChainRelayV2 = relay_mod.ValidatorChainRelayV2


class _FakeListener:
    def __init__(self, script):
        # script: list of ("raise", exc) or ("conn",) or ("block",)
        self._script = list(script)
        self._i = 0
        self.accept_calls = 0

    def accept(self):
        self.accept_calls += 1
        if self._i >= len(self._script):
            # Nothing left: block briefly so the loop doesn't busy-spin the test.
            time.sleep(0.05)
            raise BlockingIOError("no more")
        step = self._script[self._i]
        self._i += 1
        if step[0] == "raise":
            raise step[1]
        return (object(), ("cid", 0))

    def close(self):
        pass


def _run_loop(relay, listener, stop_after=1.0):
    relay._listener = listener
    t = threading.Thread(target=relay._accept_loop, daemon=True)
    t.start()
    time.sleep(stop_after)
    relay.stop()
    t.join(2)
    return t


def test_transient_accept_error_does_not_kill_the_loop(monkeypatch):
    handled = []
    monkeypatch.setattr(
        relay_mod, "handle_chain_relay_connection",
        lambda conn, connector=None: handled.append(conn),
    )
    relay = ValidatorChainRelayV2()
    # One transient error, THEN a real connection: the old code would have
    # returned on the error and never reached the connection.
    listener = _FakeListener([("raise", OSError("EMFILE")), ("conn",)])
    t = _run_loop(relay, listener)
    assert not t.is_alive()          # exited only because stop() was called
    assert len(handled) == 1         # served the connection after the error
    assert listener.accept_calls >= 2  # kept accepting past the transient error


def test_stop_exits_cleanly_even_mid_error(monkeypatch):
    monkeypatch.setattr(
        relay_mod, "handle_chain_relay_connection",
        lambda conn, connector=None: None,
    )
    relay = ValidatorChainRelayV2()
    # Listener that always raises: without stop() the loop keeps serving;
    # stop() must break it out.
    class _AlwaysRaise:
        def accept(self):
            time.sleep(0.02)
            raise OSError("ECONNABORTED")
        def close(self):
            pass
    relay._listener = _AlwaysRaise()
    t = threading.Thread(target=relay._accept_loop, daemon=True)
    t.start()
    time.sleep(0.2)
    assert t.is_alive()              # survived repeated transient errors
    relay.stop()
    t.join(2)
    assert not t.is_alive()          # stop() cleanly ended it


def test_handler_spawn_failure_does_not_kill_the_loop(monkeypatch):
    # If starting the per-connection handler thread raises (thread exhaustion),
    # the loop must close the connection and keep accepting, not die. Fail the
    # spawn ONLY for the handler target so the test's own loop thread is real.
    closed = []

    class _Conn:
        def close(self):
            closed.append(True)

    real_thread = relay_mod.threading.Thread

    class _SelectiveThread:
        def __init__(self, *a, target=None, **k):
            self._is_handler = target is relay_mod.handle_chain_relay_connection
            self._real = None if self._is_handler else real_thread(*a, target=target, **k)

        def start(self):
            if self._is_handler:
                raise RuntimeError("can't start thread")
            self._real.start()

        def join(self, *a):
            return self._real.join(*a) if self._real else None

        def is_alive(self):
            return bool(self._real and self._real.is_alive())

    monkeypatch.setattr(relay_mod.threading, "Thread", _SelectiveThread)
    relay = ValidatorChainRelayV2()

    calls = {"n": 0}

    class _OneConnThenBlock:
        def accept(self):
            calls["n"] += 1
            if calls["n"] == 1:
                return (_Conn(), ("cid", 0))
            time.sleep(0.05)
            raise BlockingIOError("idle")

        def close(self):
            pass

    relay._listener = _OneConnThenBlock()
    t = real_thread(target=relay._accept_loop, daemon=True)
    t.start()
    time.sleep(0.3)
    alive = t.is_alive()
    relay.stop()
    t.join(2)
    assert alive              # spawn failure did not kill the loop
    assert closed == [True]   # the orphaned connection was closed


class _FakeSock:
    """Minimal listener stand-in for start()/re-entrancy tests (no real vsock)."""
    open_count = 0
    close_count = 0

    def __init__(self, *_a, **_k):
        type(self).open_count += 1
        self.closed = False

    def bind(self, _addr):
        pass

    def listen(self, _backlog):
        pass

    def accept(self):
        # Block until closed so the loop thread stays alive but idle.
        while not self.closed:
            time.sleep(0.02)
        raise OSError("listener closed")

    def close(self):
        if not self.closed:
            self.closed = True
            type(self).close_count += 1


def test_start_is_reentrant_and_closes_stale_listener_on_restart(monkeypatch):
    _FakeSock.open_count = 0
    _FakeSock.close_count = 0
    relay = ValidatorChainRelayV2(socket_factory=lambda *a, **k: _FakeSock())
    relay.start()
    first_listener = relay._listener
    assert relay.status()["status"] == "running"

    # Simulate an unexpected loop-thread death WITHOUT a stop() (the case
    # main()'s supervisor must recover). The stale listener is still open.
    relay._thread = None
    assert not relay._stop.is_set()

    relay.start()  # recovery restart
    assert relay.status()["status"] == "running"
    assert relay._listener is not first_listener          # rebound fresh
    assert first_listener.closed                           # stale one closed
    assert _FakeSock.open_count == 2 and _FakeSock.close_count >= 1
    relay.stop()


def test_supervisor_restarts_dead_relay_but_respects_stop(monkeypatch):
    # Exercise the exact decision main()'s loop makes, without its infinite loop.
    relay = ValidatorChainRelayV2(socket_factory=lambda *a, **k: _FakeSock())
    relay.start()

    def supervise_once():
        if relay._stop.is_set():
            return False
        if relay.status().get("status") != "running":
            relay.start()
            return True
        return False

    # Healthy -> no restart.
    assert supervise_once() is False
    # Dead (no stop) -> restart.
    relay._thread = None
    assert supervise_once() is True
    assert relay.status()["status"] == "running"
    # Intentional stop -> supervisor must NOT resurrect it.
    relay.stop()
    relay._thread = None
    assert supervise_once() is False


def test_control_handshake_has_a_timeout_and_cleans_up_on_stall():
    # A client that connects and stalls during the control handshake must not
    # pin the handler thread/fd: the accepted connection gets a timeout, and a
    # timeout during the read is cleaned up by the finally (both sockets closed).
    import socket as _socket

    events = {"settimeouts": [], "closed": False}

    class _StalledConn:
        def settimeout(self, v):
            events["settimeouts"].append(v)

        def recv(self, _n):
            # Simulate the socket timeout firing during the handshake read.
            raise _socket.timeout("timed out")

        def sendall(self, _b):
            raise AssertionError("must not reach send on a stalled handshake")

        def close(self):
            events["closed"] = True

    # Should not hang, should not raise out (finally swallows via close), and
    # must have armed the handshake timeout before reading.
    try:
        relay_mod.handle_chain_relay_connection(_StalledConn(), connector=lambda h: None)
    except _socket.timeout:
        pass  # acceptable: propagated after cleanup
    assert events["settimeouts"], "handshake timeout was never armed"
    assert events["settimeouts"][0] == relay_mod.CONTROL_HANDSHAKE_TIMEOUT_SECONDS
    assert events["closed"] is True
