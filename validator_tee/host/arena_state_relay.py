"""Opaque vsock relay to the one Arena host in the sealed signer policy."""

from __future__ import annotations

import ipaddress
import json
import socket
import threading
from urllib.parse import urlsplit

from leadpoet_canonical.lab_arena_rewards import sha256_json
from validator_tee.enclave.arena_hotkey import validate_policy
from validator_tee.host.chain_relay_v2 import _read_control, _send_control, _relay

ARENA_STATE_RELAY_PORT = 5003
RELAY_SCHEMA = "leadpoet.arena_state_relay.v1"


class ArenaStateRelay:
    def __init__(self, policy):
        self.policy = validate_policy(policy)
        self.host = urlsplit(self.policy["arena_api_base_url"]).hostname
        self.policy_hash = sha256_json(self.policy)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(8)
        self._connections = set()
        self._threads = set()
        self._listener = None
        self._accept_thread = None

    def validate_control(self, value):
        expected = {"schema_version": RELAY_SCHEMA, "host": self.host, "port": 443, "policy_hash": self.policy_hash}
        if value != expected:
            raise ValueError("Arena state relay policy differs")

    def _connect(self):
        candidates = socket.getaddrinfo(self.host, 443, type=socket.SOCK_STREAM)
        if not candidates or any(not ipaddress.ip_address(item[4][0]).is_global for item in candidates):
            raise ValueError("Arena state relay requires a public address")
        for family, kind, protocol, _canonical, address in candidates:
            upstream = socket.socket(family, kind, protocol)
            try:
                upstream.settimeout(15)
                upstream.connect(address)
                # Bound writes even when the peer stops reading.
                upstream.settimeout(30)
                return upstream
            except OSError:
                upstream.close()
        raise ConnectionError("Arena state relay could not connect")

    def _handle(self, client):
        upstream = None
        try:
            client.settimeout(30)
            self.validate_control(_read_control(client))
            upstream = self._connect()
            with self._lock:
                if self._stop.is_set():
                    return
                self._connections.add(upstream)
            _send_control(client, {"status": "connected", "policy_hash": self.policy_hash})
            _relay(client, upstream)
        except (OSError, ValueError, RuntimeError):
            pass  # Enclave sees a closed connection and fails the read.
        finally:
            for connection in (client, upstream):
                if connection is not None:
                    with self._lock:
                        self._connections.discard(connection)
                    connection.close()
            with self._lock:
                self._threads.discard(threading.current_thread())
            self._slots.release()

    def _accept(self):
        while not self._stop.is_set():
            try:
                client, _address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            if not self._slots.acquire(blocking=False):
                client.close()
                continue
            thread = threading.Thread(target=self._handle, args=(client,), daemon=True, name="arena-state-relay-request")
            with self._lock:
                self._connections.add(client)
                self._threads.add(thread)
            thread.start()

    def start(self):
        if self._listener is not None:
            raise RuntimeError("Arena state relay already started")
        listener = socket.socket(getattr(socket, "AF_VSOCK", 40), socket.SOCK_STREAM)
        try:
            listener.settimeout(1)
            listener.bind((0xFFFFFFFF, ARENA_STATE_RELAY_PORT))
            listener.listen(8)
            self._listener = listener
            self._accept_thread = threading.Thread(target=self._accept, daemon=True, name="arena-state-relay")
            self._accept_thread.start()
        except Exception:
            listener.close()
            self._listener = None
            raise

    def stop(self):
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
        with self._lock:
            connections = list(self._connections)
            threads = list(self._threads)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        for thread in threads:
            thread.join(timeout=2)
        if any(thread.is_alive() for thread in threads) or (self._accept_thread is not None and self._accept_thread.is_alive()):
            raise RuntimeError("Arena state relay did not stop")
