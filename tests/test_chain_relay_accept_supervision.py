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
