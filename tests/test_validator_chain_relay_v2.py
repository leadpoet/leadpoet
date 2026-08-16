from __future__ import annotations

import errno
import json
import threading

import pytest

from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    chain_source_policy_hash,
)
from validator_tee.host import chain_relay_v2
from validator_tee.host.chain_relay_v2 import (
    ValidatorChainRelayV2,
    ValidatorChainRelayV2Error,
    _shutdown_and_close_socket,
    _validate_control,
    handle_chain_relay_connection,
)


def _control(**overrides):
    value = {
        "schema_version": "leadpoet.validator_chain_relay.v2",
        "host": CHAIN_ENDPOINT_HOST,
        "port": 443,
        "policy_hash": chain_source_policy_hash(),
    }
    value.update(overrides)
    return value


def test_validator_chain_relay_accepts_only_fixed_measured_destination():
    assert _validate_control(_control()) == CHAIN_ENDPOINT_HOST
    assert _validate_control(
        _control(host=CHAIN_ARCHIVE_ENDPOINT_HOST)
    ) == CHAIN_ARCHIVE_ENDPOINT_HOST
    with pytest.raises(ValidatorChainRelayV2Error, match="measured chain"):
        _validate_control(_control(host="attacker.example"))
    with pytest.raises(ValidatorChainRelayV2Error, match="measured chain"):
        _validate_control(
            _control(host="archive.chain.opentensor.ai.attacker.example")
        )
    with pytest.raises(ValidatorChainRelayV2Error, match="policy hash"):
        _validate_control(_control(policy_hash="sha256:" + "0" * 64))
    with pytest.raises(ValidatorChainRelayV2Error, match="fields"):
        _validate_control({**_control(), "extra": True})


def test_validator_chain_relay_listener_cleanup_retains_failed_ownership():
    class Listener:
        def __init__(self):
            self.close_result = False

        def shutdown(self, _how):
            return None

        def close(self):
            return self.close_result

    listener = Listener()
    assert _shutdown_and_close_socket(listener) is False
    relay = ValidatorChainRelayV2()
    relay._listener = listener

    with pytest.raises(
        ValidatorChainRelayV2Error,
        match="listener cleanup failed",
    ):
        relay.stop()

    assert relay._listener is listener
    assert relay.status()["status"] == "cleanup_failed"
    assert relay.status()["socket_cleanup_failure_count"] == 1

    listener.close_result = None
    relay.stop()
    assert relay._listener is None
    assert relay.status()["status"] == "stopped"

    class LiveThread:
        def __init__(self):
            self.joined = False

        def join(self, timeout=None):
            assert timeout == 2.0
            self.joined = True

        def is_alive(self):
            return True

    live_thread = LiveThread()
    relay._stop = threading.Event()
    relay._listener = listener
    relay._thread = live_thread
    with pytest.raises(
        ValidatorChainRelayV2Error,
        match="accept loop did not terminate",
    ):
        relay.stop()
    assert live_thread.joined is True
    assert relay._listener is listener
    assert relay._thread is live_thread
    assert relay.status()["status"] == "cleanup_failed"


def test_validator_chain_relay_start_preserves_primary_and_cleanup_failure():
    class Listener:
        def bind(self, _address):
            raise RuntimeError("primary bind failure")

        def shutdown(self, _how):
            return None

        def close(self):
            return False

    listener = Listener()
    relay = ValidatorChainRelayV2(socket_factory=lambda *_args: listener)

    with pytest.raises(
        ValidatorChainRelayV2Error,
        match="listener cleanup failed after startup",
    ) as captured:
        relay.start()

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "primary bind failure"
    assert relay._listener is listener
    assert relay.status()["last_failure"]["primary_error_type"] == "RuntimeError"


def test_validator_chain_relay_handler_cleanup_does_not_mask_primary_failure():
    callbacks = []

    class Connection:
        def recv(self, _size):
            return b""

        def shutdown(self, _how):
            return None

        def close(self):
            return False

    connection = Connection()
    with pytest.raises(
        ValidatorChainRelayV2Error,
        match="control prefix is incomplete",
    ) as captured:
        handle_chain_relay_connection(
            connection,
            cleanup_failure_callback=lambda candidate, endpoint, primary: (
                callbacks.append((candidate, endpoint, primary))
            ),
        )

    assert callbacks == [(connection, "enclave", captured.value)]


def test_validator_chain_relay_accept_loop_death_is_observable(monkeypatch):
    failures = []

    class Listener:
        def bind(self, _address):
            return None

        def listen(self, _backlog):
            return None

        def accept(self):
            raise OSError(errno.EIO, "redacted")

        def shutdown(self, _how):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        chain_relay_v2,
        "capture_failure",
        lambda *args, **kwargs: failures.append((args, kwargs)),
    )
    relay = ValidatorChainRelayV2(socket_factory=lambda *_args: Listener())
    relay.start()
    relay._thread.join(timeout=1)

    assert relay._thread.is_alive() is False
    assert relay.wait_for_accept_loop(poll_seconds=0.01) == 1
    status = relay.status()
    assert status["status"] == "failed"
    assert status["last_failure"]["stage"] == "accept_loop"
    assert status["last_failure"]["errno"] == errno.EIO
    assert failures[0][1]["stage"] == "chain_relay_accept"
    relay.stop()


def test_validator_chain_relay_main_exits_nonzero_when_accept_loop_dies(
    monkeypatch,
):
    observed = []

    class Relay:
        def start(self):
            return {"status": "running"}

        def wait_for_accept_loop(self):
            return 1

        def stop(self):
            observed.append("stop")

    monkeypatch.setattr(chain_relay_v2, "ValidatorChainRelayV2", Relay)
    monkeypatch.setattr(
        chain_relay_v2,
        "capture_failure",
        lambda *args, **kwargs: observed.append(kwargs["stage"]),
    )

    assert chain_relay_v2.main() == 1
    assert observed == ["stop"]
