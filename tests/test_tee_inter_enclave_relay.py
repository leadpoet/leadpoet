import json
import socket
import threading

from gateway.tee.topology import topology_hash
from gateway.utils.tee_client import _recv_exact
from gateway.utils.tee_inter_enclave_relay import _handle_connection


def _frame(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return len(encoded).to_bytes(4, "big") + encoded


def _read_frame(connection):
    prefix = _recv_exact(connection, 4)
    body = _recv_exact(connection, int.from_bytes(prefix, "big"))
    return json.loads(body.decode("ascii"))


def _request(*, source=17, target=16):
    return {
        "schema_version": "leadpoet.inter_enclave_relay.v2",
        "channel_id": "a" * 32,
        "source_cid": source,
        "target_cid": target,
        "target_port": 5003,
        "topology_hash": topology_hash(),
    }


def test_parent_relay_forwards_only_opaque_channel_bytes():
    runner, parent = socket.socketpair()
    coordinator, target = socket.socketpair()
    connected = []

    def connector(cid, port):
        connected.append((cid, port))
        return coordinator

    thread = threading.Thread(
        target=_handle_connection,
        kwargs={
            "connection": parent,
            "source_cid": 17,
            "connector": connector,
            "idle_timeout": 2.0,
        },
        daemon=True,
    )
    thread.start()
    try:
        runner.sendall(_frame(_request()))
        response = _read_frame(runner)
        assert response["result"]["status"] == "connected"
        assert connected == [(16, 5003)]
        runner.sendall(b"opaque-tls-application-record")
        assert target.recv(64) == b"opaque-tls-application-record"
        target.sendall(b"opaque-tls-response-record")
        assert runner.recv(64) == b"opaque-tls-response-record"
    finally:
        runner.close()
        target.close()
        thread.join(timeout=2)


def test_parent_relay_rejects_runner_to_runner_channels():
    runner, parent = socket.socketpair()
    called = []

    def connector(cid, port):
        called.append((cid, port))
        raise AssertionError("must not connect")

    thread = threading.Thread(
        target=_handle_connection,
        kwargs={"connection": parent, "source_cid": 17, "connector": connector},
        daemon=True,
    )
    thread.start()
    runner.sendall(_frame(_request(source=17, target=18)))
    response = _read_frame(runner)
    runner.close()
    thread.join(timeout=2)
    assert response["status"] == "error"
    assert called == []


def test_parent_relay_rejects_spoofed_source_cid():
    runner, parent = socket.socketpair()
    thread = threading.Thread(
        target=_handle_connection,
        kwargs={"connection": parent, "source_cid": 19},
        daemon=True,
    )
    thread.start()
    runner.sendall(_frame(_request(source=17, target=16)))
    response = _read_frame(runner)
    runner.close()
    thread.join(timeout=2)
    assert response["status"] == "error"
