"""TLS-authenticated accepted Arena state reads through the host byte relay."""

from __future__ import annotations

import http.client
import json
import os
import socket
import ssl
import threading
from typing import Any, Dict, Mapping

from leadpoet_canonical.lab_arena_rewards import canonical_json, sha256_json

AF_VSOCK = 40
PARENT_CID = 3
ARENA_STATE_RELAY_PORT = 5003
MAX_CONTROL_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 30
DEFAULT_CA_BUNDLE = "/etc/pki/tls/certs/ca-bundle.crt"


class ArenaStateSourceError(RuntimeError):
    pass


def _recv_exact(connection: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(size - len(output))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


class ArenaStateSource:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        from urllib.parse import urlsplit
        self._policy = json.loads(json.dumps(dict(policy)))
        endpoint = urlsplit(self._policy["arena_api_base_url"])
        self._host = str(endpoint.hostname)
        self._policy_hash = sha256_json(self._policy)
        self._lock = threading.Lock()

    def read(self, *, epoch: int) -> Dict[str, Any]:
        normalized_epoch = int(epoch)
        if normalized_epoch < 0:
            raise ArenaStateSourceError("Arena state epoch is invalid")
        with self._lock:
            parent = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
            response = None
            try:
                parent.settimeout(TIMEOUT_SECONDS)
                parent.connect((PARENT_CID, ARENA_STATE_RELAY_PORT))
                control = canonical_json({
                    "schema_version": "leadpoet.arena_state_relay.v1",
                    "host": self._host, "port": 443, "policy_hash": self._policy_hash,
                }).encode("ascii")
                parent.sendall(len(control).to_bytes(4, "big") + control)
                prefix = _recv_exact(parent, 4)
                if len(prefix) != 4:
                    raise ArenaStateSourceError("Arena state relay response is incomplete")
                size = int.from_bytes(prefix, "big")
                if not 2 <= size <= MAX_CONTROL_BYTES:
                    raise ArenaStateSourceError("Arena state relay response size is invalid")
                reply = json.loads(_recv_exact(parent, size).decode("ascii"))
                if reply != {"status": "connected", "policy_hash": self._policy_hash}:
                    raise ArenaStateSourceError("Arena state relay refused sealed policy")
                if not os.path.isfile(DEFAULT_CA_BUNDLE):
                    raise ArenaStateSourceError("measured CA bundle is unavailable")
                context = ssl.create_default_context(cafile=DEFAULT_CA_BUNDLE)
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                tls = context.wrap_socket(parent, server_hostname=self._host)
                parent = None
                path = "/arena/v1/weight-state?epoch=%d" % normalized_epoch
                tls.sendall((
                    "GET %s HTTP/1.1\r\nHost: %s\r\nAccept: application/json\r\n"
                    "Accept-Encoding: identity\r\nConnection: close\r\n\r\n"
                    % (path, self._host)
                ).encode("ascii"))
                response = http.client.HTTPResponse(tls)
                response.begin()
                if int(response.status) != 200:
                    raise ArenaStateSourceError("Arena state endpoint returned HTTP %d" % int(response.status))
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ArenaStateSourceError("Arena state response exceeds limit")
                value = json.loads(body.decode("utf-8"))
                if not isinstance(value, dict) or set(value) != {"lookup_ok", "state"} or value.get("lookup_ok") is not True or not isinstance(value.get("state"), dict):
                    raise ArenaStateSourceError("Arena state response is invalid")
                return dict(value["state"])
            except ArenaStateSourceError:
                raise
            except Exception as exc:
                raise ArenaStateSourceError("authenticated Arena state read failed") from exc
            finally:
                if response is not None:
                    response.close()
                connection = locals().get("tls") or parent
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    connection.close()
